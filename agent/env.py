import carla
import gym
import time
import random
import numpy as np
import math
from gym import spaces
from absl import logging
from collections import deque
import pygame
import cv2


from utils.graphics import HUD
from utils.utils import get_actor_display_name, smooth_action, vector, distance_to_line, build_projection_matrix, get_image_point
from agent.actions import CarlaActions
from agent.observation import CarlaObservations
from utils.planner import compute_route_waypoints

logging.set_verbosity(logging.INFO)

# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, host, port, town, fps, obs_sensor, obs_res, view_res, reward_fn, action_smoothing, allow_render=True, allow_spectator=True):
        
        self.obs_width, self.obs_height = obs_res
        self.spectator_width, self.spectator_height = view_res
        self.allow_render = allow_render
        self.allow_spectator = allow_spectator
        self.spectator_camera = None
        self.episode_idx = -2
        self.world = None
        self.actions = CarlaActions()
        self.observations = CarlaObservations(self.obs_height, self.obs_width)
        self.obs_sensor = obs_sensor
        self.control = carla.VehicleControl()
        self.action_space = self.actions.get_action_space()
        self.observation_space = self.observations.get_observation_space()
        self.max_distance = 3000
        self.action_smoothing = action_smoothing
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn 

        try:
            self.client = carla.Client(host, port)  
            self.client.set_timeout(100.0)

            self.client.load_world(map_name=town)
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)  
            self.world.apply_settings(
                carla.WorldSettings(  
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
            self.client.reload_world(False)  # reload map keeping the world settings

            # Spawn Vehicle
            self.tesla = self.world.blueprint_library.filter('model3')[0]
            self.start_transform = self._get_start_transform()
            self.curr_loc = self.start_transform.location
            self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)

           # Spawn collision and Lane invasion sensors
            colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
            lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
            self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
            self.colsensor.listen(self._collision_data)
            self.lanesensor.listen(self._lane_invasion_data)  

            # Create hud and initialize pygame for visualization
            if self.allow_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width, self.spectator_height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            # Set observation image
            if 'rgb' in self.obs_sensor:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            elif 'semantic' in self.sensors:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            else:
                raise NotImplementedError('unknown sensor type')

            self.rgb_cam.set_attribute('image_size_x', f'{self.obs_width}')
            self.rgb_cam.set_attribute('image_size_y', f'{self.obs_height}')
            self.rgb_cam.set_attribute('fov', '90')

            bound_x = self.vehicle.bounding_box.extent.x
            transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
            self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
            self.sensor_front.listen(self._set_observation_image)          
 
            # Set spectator cam   
            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', '800')
                self.spectator_camera.set_attribute('image_size_y', '800')
                self.spectator_camera.set_attribute('fov', '100')
                transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
                self.spectator_sensor.listen(self._set_viewer_image)
                    
        except RuntimeError as msg:
            pass

        self.reset()


    # Resets environment for new episode
    def reset(self):

        self.episode_idx += 1
        self.num_routes_completed = -1
        # Generate a random route
        self.generate_route()

        self.terminate = False
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0

        # reset metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.world.tick()

        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs

    def generate_route(self):
        # Do a soft reset (teleport vehicle)
        self.control.steer = float(0.0)
        self.control.brake = float(0.0)
        self.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)  # Reset the car's physics

        # Generate waypoints along the lap

        spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

    # Steps environment
    def step(self, action):

        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                self.success_state = True

            throttle, brake, steer = [float(a) for a in action]

            # Perfom action
            self.control.throttle = smooth_action(self.control.throttle, throttle, self.action_smoothing)
            self.control.brake = smooth_action(self.control.brake, brake, self.action_smoothing)
            self.control.steer = smooth_action(self.control.steer, steer, self.action_smoothing)

            self.vehicle.apply_control(self.control)

        self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        if self.allow_spectator:
            self.viewer_image = self._get_viewer_image()
            
        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)
        
        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        self.step_count += 1

        if self.allow_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminate = True
            self.render()

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }

        return self.observation, self.last_reward, self.terminate or self.success_state, info
    
    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()

        self.closed = True
    
    def render(self, mode="human"):
 
        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)
  
        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.allow_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.spectator_camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list
        # Render to screen
        pygame.display.flip()

    def _destroy_agents(self):

        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):

        # What we collided with and what was the impulse
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminate = True
        if self.allow_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

        #collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        
    def _lane_invasion_data(self, event):

        self.terminate = True
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.allow_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_start_transform(self):
        return random.choice(self.map.get_spawn_points())  

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image         
            
    
        