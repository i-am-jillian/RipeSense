#!/usr/bin/env python3
# encoding: utf-8
"""Sort fruit with the Hiwonder AiNex + RipeSense: ripe -> basket, spoiled -> bin.

Adapted from Hiwonder's ainex_example autonomous_transport_node.py (same gait,
approach thresholds and pick/place actions), with two changes:
  * the target is the best pickable fruit from fruit_detector_node
    (/fruit_detection/target) instead of a coloured block;
  * two drop zones marked by 10 cm AprilTags (tag36h11): the basket
    (~basket_tag, id 1) for ripe fruit and the bin (~bin_tag, id 2) for
    spoiled fruit. The detector's label says which ("basket:ripe_banana",
    "bin:spoiled_banana"); it is locked in at the moment of the pick.
    Unripe fruit is never targeted, so it stays where it is.

Cycle: search fruit -> approach -> 'move_up' (pick) -> search basket/bin tag ->
approach -> 'put_down' -> repeat until ~max_fruits or nothing found for
~give_up_s seconds.
"""
import json
import math
import signal

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import String
from ainex_sdk import pid, common
from ainex_example.color_common import Common
from ainex_example.pid_track import PIDTrack
from ainex_example.approach_object import ApproachObject
from apriltag_ros.msg import AprilTagDetectionArray
from ainex_interfaces.msg import ObjectsInfo
import two_hand_grab


def call(srv):
    try:
        rospy.wait_for_service(srv, timeout=3)
        rospy.ServiceProxy(srv, Empty)()
    except Exception as e:  # noqa: BLE001
        rospy.logwarn('service %s failed: %s', srv, e)


class FruitPickNode(Common):
    fruit_x_stop = 320 / 640   # stop when fruit centre is at this x fraction
    fruit_y_stop = 220 / 480   # ... and this y fraction (between the hands)
    tag_y_stop = 240 / 480
    yaw_stop_value = 0

    pick_action = 'move_up'
    place_action = 'put_down'

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.count_miss = 0
        self.object_info = None
        self.start_index = 0
        self.basket_tag = rospy.get_param('~basket_tag', 1)
        self.bin_tag = rospy.get_param('~bin_tag', 2)
        self.target_action = None     # action of the fruit currently tracked
        self.target_label = ''
        self.carrying = None          # 'basket' or 'bin' once picked up
        self.carrying_label = ''
        self.place_tag = self.basket_tag
        self.delivered = {'basket': 0, 'bin': 0}
        self.max_fruits = rospy.get_param('~max_fruits', 3)
        self.give_up_s = rospy.get_param('~give_up_s', 60.0)
        self.search_turn = rospy.get_param('~search_turn_deg', -8)   # per step after an empty head sweep; negative = right
        self.search_steps = rospy.get_param('~search_steps', 2)       # bounded turn, never continuous
        self.search_dwell = rospy.get_param('~search_dwell_s', 0.8)   # pause at each head pose so the detector can look
        self.halt_after_s = rospy.get_param('~halt_after_s', 0.4)     # target missing this long -> stop walking
        self.lost_wait_s = rospy.get_param('~lost_wait_s', 3.0)       # ... and this long -> start searching
        self.squeeze = rospy.get_param('~squeeze', 110)                # two-hand grip: 0..110 (110 = Hiwonder block grip)
        self.align_timeout = rospy.get_param('~align_timeout_s', 8.0)
        self.align_tol_px = rospy.get_param('~align_tol_px', 70)
        self.align_start = 0.0
        self.picked = 0
        self.y_stop = self.fruit_y_stop
        self.body_track_state = 'approach'
        self.head_track_state = 'track_fruit'
        self.state = 'pick'            # 'pick' fruit or 'place' in basket
        self.start_find = False
        self.last_seen = rospy.get_time()

        self.head_pan_init = 500
        self.head_tilt_init = 280
        self.head_time_stamp = rospy.get_time()
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/ainex_example/config/calib.yaml')
        self.approach_object = ApproachObject(self.gait_manager)
        self.approach_object.update_stop_count(1)
        self.approach_object.update_approach_stop_value(10, 40, 7)
        self.approach_object.update_gait_range(x_range=[-0.015, 0.015])

        self.walking_param = self.gait_manager.get_gait_param()
        self.walking_param['step_height'] = 0.025
        self.walking_param['pelvis_offset'] = 3
        self.walking_param['hip_pitch_offset'] = 15
        self.walking_param['body_height'] = 0.035
        self.approach_object.update_gait(walking_param=self.walking_param)

        self.motion_manager.run_action('walk_ready')
        signal.signal(signal.SIGINT, self.shutdown)

        self.head_pan_range = [125, 875]
        self.head_tilt_range = [250, 500]
        self.rl_dis = None
        self.ud_dis = None
        self.rl_track = PIDTrack(pid.PID(0.1, 0.0, 0.001), self.head_pan_range, self.head_pan_init)
        self.ud_track = PIDTrack(pid.PID(0.1, 0.0, 0.001), self.head_tilt_range, self.head_tilt_init)

        # head search poses [pan, tilt, ms]: left_down, left_up, centre_up, right_up, right_down
        self.head_search_fruit = [[650, 300, 1000], [650, 500, 1000], [500, 500, 1000],
                                  [350, 500, 1000], [350, 300, 1000]]
        self.head_search_tag = [[550, 300, 1000], [550, 500, 1000], [500, 500, 1000],
                                [450, 500, 1000], [450, 300, 1000]]
        self.head_move_position = self.head_search_fruit

        self.status_pub = rospy.Publisher('/fruit_pick/status', String, queue_size=1, latch=True)
        self.phase = 'starting'
        self.set_phase('searching for fruit to sort')
        call('/apriltag_ros/stop')
        call('/fruit_detection/start')
        rospy.Subscriber('/fruit_detection/target', ObjectsInfo, self.fruit_callback, queue_size=1)
        rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback, queue_size=1)

        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('fruit pick: start (max %d fruits, basket tag %d)' % (self.max_fruits, self.basket_tag))

    def set_phase(self, phase):
        self.phase = phase
        self.status_pub.publish(String(json.dumps({
            'state': self.state, 'phase': phase, 'picked': self.picked, 'delivered': self.delivered,
            'carrying': self.carrying, 'carrying_label': self.carrying_label, 'place_tag': self.place_tag,
            'max_fruits': self.max_fruits, 'basket_tag': self.basket_tag, 'bin_tag': self.bin_tag,
            't': rospy.get_time()})))
        common.loginfo('[fruit_pick] ' + phase)

    # Common's enter/start/stop talk to /color_detection; use the fruit detector instead.
    def enter_func(self, msg):
        self.init_action(self.head_pan_init, self.head_tilt_init)
        return []

    def start_srv_callback(self, msg):
        with self.lock:
            call('/fruit_detection/start' if self.state == 'pick' else '/apriltag_ros/start')
            self.start = True
        return []

    def stop_srv_callback(self, msg):
        with self.lock:
            self.start = False
            call('/fruit_detection/stop')
            call('/apriltag_ros/stop')
            self.init_action(self.head_pan_init, self.head_tilt_init)
        return []

    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            common.loginfo('%s shutdown' % self.name)

    # ---------- perception callbacks ----------
    def fruit_callback(self, msg):
        if self.state != 'pick':
            return
        for o in msg.data:
            action = o.label.split(':', 1)[0]
            if action not in ('basket', 'bin'):
                continue
            self.target_action, self.target_label = action, o.label.split(':', 1)[-1]
            self.object_info = [o.x, o.y, o.angle, o.width, o.height]
            self.last_seen = rospy.get_time()
            if self.head_track_state != 'stop' and not self.start_find:
                self.rl_dis, self.ud_dis = self.head_track_process(o.x, o.y, o.width, o.height)
            elif self.head_track_state == 'stop':
                self.rl_dis, self.ud_dis = None, None

    def tag_callback(self, msg):
        if self.state != 'place':
            return
        for d in msg.detections:
            if d.id[0] != self.place_tag:
                continue
            _, _, yaw = common.qua2rpy(d.pose.pose.pose.orientation)
            x = int((d.corners[0].x + d.corners[2].x) / 2)
            y = int((d.corners[0].y + d.corners[2].y) / 2)
            self.object_info = [x, y, math.degrees(yaw), 640, 480]
            self.last_seen = rospy.get_time()
            if self.head_track_state != 'stop' and not self.start_find:
                self.rl_dis, self.ud_dis = self.head_track_process(x, y, 640, 480)
            elif self.head_track_state == 'stop':
                self.rl_dis, self.ud_dis = None, None

    # ---------- state switches ----------
    def go_pick(self):
        self.state = 'pick'
        self.y_stop = self.fruit_y_stop
        self.head_track_state = 'track_fruit'
        self.approach_object.update_approach_stop_value(10, 40, 7)
        self.walking_param['hip_pitch_offset'] = 15
        self.approach_object.update_gait([400, 0.2, 0.025], 30, self.walking_param)
        self.head_move_position = self.head_search_fruit
        call('/apriltag_ros/stop')
        call('/fruit_detection/start')
        self.object_info = None
        self.last_seen = rospy.get_time()
        self.set_phase('searching for fruit to sort')

    def go_place(self):
        self.state = 'place'
        self.y_stop = self.tag_y_stop
        self.head_track_state = 'track_tag'
        self.approach_object.update_approach_stop_value(10, 10, 5)
        self.walking_param['hip_pitch_offset'] = 5
        self.approach_object.update_gait([400, 0.1, 0.03], 0, self.walking_param)
        self.head_move_position = self.head_search_tag
        call('/fruit_detection/stop')
        call('/apriltag_ros/start')
        self.object_info = None
        self.last_seen = rospy.get_time()
        self.place_tag = self.basket_tag if self.carrying == 'basket' else self.bin_tag
        self.set_phase('carrying %s, looking for %s (tag %d)' % (self.carrying_label.replace('_', ' '), self.carrying, self.place_tag))

    # ---------- tracking (unchanged from Hiwonder) ----------
    def head_track_process(self, x, y, width, height):
        if abs(x - width / 2) < 10:
            x = width / 2
        if abs(y - height / 2) < 10:
            y = height / 2
        rl_dis = self.rl_track.track(x, width / 2)
        ud_dis = self.ud_track.track(y, height / 2)
        self.motion_manager.set_servos_position(20, [[23, int(rl_dis)], [24, int(ud_dis)]])
        return rl_dis, ud_dis

    def body_track_process(self, rl_dis, ud_dis, object_data):
        self.object_info = None
        x, y, angle, width, height = object_data
        self.rl_dis, self.ud_dis = None, None
        if self.body_track_state == 'approach' and rl_dis is not None:
            want = 'walking to ' + (self.target_label.replace('_', ' ') if self.state == 'pick' else self.carrying)
            if self.phase != want:
                self.set_phase(want)
            self.approach_object.update_gait(step_mode=0)
            self.approach_object.update_gait_range(y_range=[-0.01, 0.01], yaw_range=[-6, 6])
            yaw_stop = 500 + self.calib_config['head_pan_offset'] + math.copysign(
                self.yaw_stop_value, rl_dis - (500 + self.calib_config['head_pan_offset']))
            if self.approach_object.process(500 - ud_dis, 0 + self.calib_config['center_x_offset'],
                                            (yaw_stop - rl_dis) / 9, self.head_tilt_range[0] - 40,
                                            0, 0, width, height):
                self.head_track_state = 'stop'
                self.body_track_state = 'align'
                self.align_start = rospy.get_time()
                rospy.sleep(0.5)
                self.motion_manager.set_servos_position(200, [[23, self.head_pan_init], [24, self.head_tilt_init]])
                rospy.sleep(0.5)
                self.set_phase('lining up with %s' % ('fruit' if self.state == 'pick' else self.carrying))
        elif self.body_track_state == 'align':
            if self.state == 'pick':
                pass                            # detector sends 0 for round fruit, long-axis angle for bananas
            else:
                self.walking_param['hip_pitch_offset'] = 5
                self.approach_object.update_gait([400, 0.1, 0.03], 0, self.walking_param)
                if angle < 0:
                    angle += 360
                angle -= 180
            self.approach_object.update_gait(step_mode=1)
            self.approach_object.update_gait_range(y_range=[-0.012, 0.012], yaw_range=[-4, 4])
            aligned = self.approach_object.process(y, x, angle, self.y_stop * width, self.fruit_x_stop * width,
                                                   0, width, height)
            # The stock fine-alignment can shuffle back and forth for minutes when the fruit is large in
            # view. After align_timeout_s, grab anyway if the fruit is roughly between the feet.
            if not aligned and self.state == 'pick' and rospy.get_time() - self.align_start > self.align_timeout and \
                    abs(x - self.fruit_x_stop * width) < self.align_tol_px and abs(y - self.y_stop * height) < self.align_tol_px:
                common.loginfo('align timeout: fruit within %d px, grabbing' % self.align_tol_px)
                self.gait_manager.stop()
                aligned = True
            if aligned:
                if self.state == 'pick':
                    self.gait_manager.disable()
                    self.carrying, self.carrying_label = self.target_action or 'basket', self.target_label
                    self.set_phase('picking up %s -> %s' % (self.carrying_label.replace('_', ' '), self.carrying))
                    two_hand_grab.grab(self.motion_manager, self.squeeze)      # both hands, sized for the fruit
                else:
                    self.walking_param['hip_pitch_offset'] = 5
                    self.gait_manager.set_step([400, 0.2, 0.025], 0.01, 0, 0, self.walking_param, 0, 2)
                    self.gait_manager.disable()
                    self.set_phase('dropping %s in the %s' % (self.carrying_label.replace('_', ' '), self.carrying))
                    two_hand_grab.release(self.motion_manager, self.squeeze)
                    self.walking_param['hip_pitch_offset'] = 15
                    self.gait_manager.set_step([400, 0.2, 0.025], -0.01, 0, 0, self.walking_param, 0, 7)
                self.body_track_state = 'approach'
                return True
            else:
                rospy.sleep(0.8)
        return False

    def find_process(self):
        """Search without wandering: sweep the head through the search poses with the body
        still, pausing at each so the detector can look; only after a full empty sweep turn
        the body a fixed, small number of steps (the stock code started a continuous turn)."""
        if rospy.get_time() > self.head_time_stamp:
            if self.start_index > len(self.head_move_position) - 1:
                self.start_index = 0
                self.set_phase('nothing in view: turning %d steps' % self.search_steps)
                self.walking_param['hip_pitch_offset'] = 5 if self.state == 'place' else 15
                self.gait_manager.set_step([400, 0.2, 0.025], 0, 0, self.search_turn, self.walking_param, 0,
                                           self.search_steps)          # step_num > 0: blocks, then stops
                self.gait_manager.stop()
                self.set_phase('searching for ' + ('fruit to sort' if self.state == 'pick' else
                                                   '%s tag %d' % (self.carrying, self.place_tag)))
            pan, tilt, ms = self.head_move_position[self.start_index]
            self.rl_track.update_position(pan)
            self.ud_track.update_position(tilt)
            self.motion_manager.set_servos_position(ms, [[23, pan], [24, tilt]])
            self.head_time_stamp = rospy.get_time() + ms / 1000.0 + self.search_dwell
            self.start_index += 1

    def run(self):
        halted = False
        while self.running and not rospy.is_shutdown():
            if self.start:
                if self.object_info is not None:
                    halted = False
                    if self.start_find:
                        self.start_find = False
                        self.gait_manager.stop()
                    elif self.body_track_process(self.rl_dis, self.ud_dis, self.object_info):
                        if self.state == 'pick':
                            self.go_place()
                        else:
                            self.picked += 1
                            self.delivered[self.carrying or 'basket'] += 1
                            self.carrying, self.carrying_label = None, ''
                            common.loginfo('fruit %d/%d delivered' % (self.picked, self.max_fruits))
                            if self.picked >= self.max_fruits:
                                break
                            self.go_pick()
                else:
                    lost = rospy.get_time() - self.last_seen
                    if lost > self.give_up_s:
                        common.loginfo('nothing found for %.0fs while in %s state; stopping' % (self.give_up_s, self.state))
                        break
                    if not self.start_find:
                        if lost > self.halt_after_s and not halted:
                            # lost it mid-walk: stop the legs NOW (a continuous gait command would keep going),
                            # keep the head where it last saw the fruit and wait for it to reappear
                            self.gait_manager.stop()
                            halted = True
                            self.set_phase('lost sight of it: standing still, looking')
                        if lost > self.lost_wait_s:
                            self.start_find = True
                            self.start_index = 0
                            self.head_time_stamp = 0
                            self.set_phase('searching for ' + ('fruit to sort' if self.state == 'pick' else
                                                               '%s tag %d' % (self.carrying, self.place_tag)))
                    else:
                        self.find_process()
            rospy.sleep(0.01)

        self.gait_manager.stop()
        self.init_action(self.head_pan_init, self.head_tilt_init)
        call('/fruit_detection/stop')
        call('/apriltag_ros/stop')
        self.set_phase('finished: %d ripe to basket, %d spoiled to bin' % (self.delivered['basket'], self.delivered['bin']))
        rospy.signal_shutdown('done')


if __name__ == '__main__':
    FruitPickNode('fruit_pick').run()
