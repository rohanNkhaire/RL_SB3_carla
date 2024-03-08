import os
import signal
import sys
import carla
import gym
import time
import random
import numpy as np
import math
from queue import Queue
from gym import spaces
from absl import logging
import carla_utils.graphics
import pygame
import atexit
from subprocess import check_output

from core_rl.actions import CarlaActions
from core_rl.observation import CarlaObservations

logging.set_verbosity(logging.INFO)

# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, town, fps, img_width, img_height, repeat_action, start_transform_type, sensors,
                 action_type, enable_preview, steps_per_episode, playing=False, timeout=60):

        try:
            client = carla.Client("localhost", 2000)  
            client.set_timeout(100.0)

            client.load_world(map_name=town)
            self.world = client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)  
            frame = self.world.apply_settings(
                carla.WorldSettings(  
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
        except RuntimeError as msg:
            pass

        self.server = self.get_pid("CarlaUE4-Linux-Shipping")    
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.tesla = blueprint_library.filter('tesla')[0]
        self.img_width = img_width
        self.img_height = img_height
        self.repeat_action = repeat_action
        self.start_transform_type = start_transform_type
        self.sensors = sensors
        self.actor_list = []
        self.preview_camera = None
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview
        self.observation = CarlaObservations(img_height, img_width)
        self.actions = CarlaActions(action_type)

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation space of the sensor."""
        return self.observation.get_observation_space()

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        return self.actions.get_action_space()


    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed

    # Resets environment for new episode
    def reset(self):
        self._destroy_agents()
        # logging.debug("Resetting environment")
        # Car, sensors, etc. We create them every episode then destroy
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0
        # self.total_reward = 0

        self.front_image_Queue = Queue()
        self.preview_image_Queue = Queue()

        # self.episode += 1

        # When Carla breaks (stopps working) or spawn point is already occupied, spawning a car throws an exception
        # We allow it to try for 3 seconds then forgive
        spawn_start = time.time()
        while True:
            try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                self.start_transform = self._get_start_transform()
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)
                break
            except Exception as e:
                logging.error('Error carla 141 {}'.format(str(e)))
                time.sleep(0.01)

            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        # Append actor to a list of spawned actors, we need to remove them later
        self.actor_list.append(self.vehicle)

        # TODO: combine the sensors
        if 'rgb' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif 'semantic' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            raise NotImplementedError('unknown sensor type')

        self.rgb_cam.set_attribute('image_size_x', f'{self.img_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.img_height}')
        self.rgb_cam.set_attribute('fov', '90')

        bound_x = self.vehicle.bounding_box.extent.x
        bound_y = self.vehicle.bounding_box.extent.y


        transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.front_image_Queue.put)
        self.actor_list.extend([self.sensor_front])

        # Preview ("above the car") camera
        if self.preview_camera_enabled:
            # TODO: add the configs
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', '400')
            self.preview_cam.set_attribute('image_size_y', '400')
            self.preview_cam.set_attribute('fov', '100')
            transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)

        # Here's some workarounds.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        # Collision history is a list callback is going to append to (we brake simulation on a collision)
        self.collision_hist = []
        self.lane_invasion_hist = []

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.lanesensor.listen(self._lane_invasion_data)
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)

        self.world.tick()

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.front_image_Queue.empty():
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, -1))
        image = image[:, :, :3]

        return image

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info

    # Steps environment
    def _step(self, action):
        self.world.tick()
        self.render()
            
        self.frame_step += 1

        # Apply control to the vehicle based on an action
        self.vehicle.apply_control(self.actions.action_to_control(action))

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, -1))

        # TODO: Combine the sensors
        if 'rgb' in self.sensors:
            image = image[:, :, :3]
        if 'semantic' in self.sensors:
            image = image[:, :, 2]
            image = (np.arange(13) == image[..., None])
            image = np.concatenate((image[:, :, 2:3], image[:, :, 6:8]), axis=2)
            image = image * 255

        done = False
        reward = 0
        info = dict()

        # # If car collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward += -100
            self.collision_hist = []
            self.lane_invasion_hist = []

        if len(self.lane_invasion_hist) != 0:
            reward += -5
            self.lane_invasion_hist = []

        # # Reward for speed
        # if not self.playing:
        #     reward += 0.1 * kmh * (self.frame_step + 1)
        # else:
        #     reward += 0.1 * kmh

        reward += 0.1 * kmh

        reward += square_dist_diff

        # # Reward for distance to road lines
        # if not self.playing:
        #     reward -= math.exp(-dis_to_left)
        #     reward -= math.exp(-dis_to_right)
        
        if self.frame_step >= self.steps_per_episode:
            done = True

        if not self._on_highway():
            self.out_of_loop += 1
            if self.out_of_loop >= 20:
                done = True
        else:
            self.out_of_loop = 0

        # self.total_reward += reward

        if done:
            # info['episode'] = {}
            # info['episode']['l'] = self.frame_step
            # info['episode']['r'] = reward
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()
        
        return image, reward, done, info
    
    def close(self):
        '''
        logging.info("Closes the CARLA server with process PID {}".format(self.server))
        os.killpg(os.getpgid(self.server), signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(os.getpgid(self.server), signal.SIGKILL))
        '''
        pass
    
    def render(self):
        # TODO: clean this
        # TODO: change the width and height to compat with the preview cam config

        if self.preview_camera_enabled:

            self._display, self._clock, self._font = carla_utils.graphics.setup(
                width=400,
                height=400,
                render=True,
            )
            mode = 'human'
            preview_img = self.preview_image_Queue.get()
            preview_img = np.array(preview_img.raw_data)
            preview_img = preview_img.reshape((400, 400, -1))
            preview_img = preview_img[:, :, :3]
            carla_utils.graphics.make_dashboard(
                display=self._display,
                font=self._font,
                clock=self._clock,
                observations={"preview_camera":preview_img},
            )

            if mode == "human":
                # Update window display.
                pygame.display.flip()
            else:
                raise NotImplementedError()

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
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # # Filter collisions
        # for actor_id, impulse in COLLISION_FILTER:
        #     if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
        #         return

        # Add collision
        self.collision_hist.append(event)
    
    def _lane_invasion_data(self, event):
        # Change this function to filter lane invasions
        self.lane_invasion_hist.append(event)

    def _on_highway(self):
        goal_abs_lane_id = 4
        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
        assert lane_id_sign in [-1, 1]
        goal_lane_id = goal_abs_lane_id * lane_id_sign
        vehicle_s = vehicle_waypoint_closest_to_road.s
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        return not (goal_waypoint is None)

    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())
        if self.start_transform_type == 'highway':
            if self.map.name == "Town04":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)): # TODO: change this
                        break
                return start_transform
            else:
                raise NotImplementedError
            
    def get_pid(self, name):
        return int(check_output(["pidof", "-s", name]))        