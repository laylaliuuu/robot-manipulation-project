import pybullet as p
import numpy as np
import cv2

class PerceptionModule:
    def __init__(self, robot_id, object_map):
        """
        object_map: dict mapping object names to their PyBullet IDs
        Example: {'red_block': 3, 'green_cube': 4, 'blue_sphere': 5}
        """
        self.robot_id = robot_id
        self.object_map = object_map
        self.camera_config = {
            'width': 640,
            'height': 480,
            'eye_pos': [0.5, 0.5, 1.5],
            'target_pos': [0.5, 0, 0.5],
            'up_vector': [0, 0, 1]
        }
    
    def get_object_position(self, object_name):
        """
        Simulator Query
        Directly query object position from PyBullet
        """
        if object_name not in self.object_map:
            return None
        
        obj_id = self.object_map[object_name]
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return np.array(pos)
    
    def render_scene_image(self):
        """
        Vision-Based
        Capture RGB image from virtual camera
        """
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_config['eye_pos'],
            cameraTargetPosition=self.camera_config['target_pos'],
            cameraUpVector=self.camera_config['up_vector']
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_config['width'] / self.camera_config['height'],
            nearVal=0.01,
            farVal=100
        )
        
        img_width, img_height, rgba, depth, segmentation = p.getCameraImage(
            self.camera_config['width'],
            self.camera_config['height'],
            view_matrix,
            proj_matrix
        )
        
        # Convert RGBA to BGR for OpenCV
        rgb = rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    
    def detect_color_objects(self, image):
        """
        Detect colored objects in image using HSV color space
        Returns dict of {color: (cx, cy)} pixel coordinates
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # HSV color ranges
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
        }
        
        detections = {}
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections[color_name] = (cx, cy)
        
        return detections
    
    def pixel_to_world(self, cx, cy, depth=0.1):
        """
        Convert 2D pixel coordinates to 3D world coordinates
        This is a simplified version - real implementation needs camera calibration
        
        For now, we'll map image coordinates to a plane at fixed depth
        """
        # Image dimensions
        img_w = self.camera_config['width']
        img_h = self.camera_config['height']
        
        # Camera intrinsics (simplified)
        fov = 60
        f = (img_w / 2) / np.tan(np.radians(fov / 2))
        
        # Normalized coordinates [-1, 1]
        x_norm = (cx - img_w / 2) / f * depth
        y_norm = (cy - img_h / 2) / f * depth
        
        # World coordinates (simplified)
        camera_pos = np.array(self.camera_config['eye_pos'])
        target_pos = np.array(self.camera_config['target_pos'])
        
        # Approximate world position
        world_x = target_pos[0] + x_norm
        world_y = target_pos[1] + y_norm
        world_z = depth
        
        return np.array([world_x, world_y, world_z])
    
    def detect_object(self, object_name):
        """
        Main detection function
        Returns 3D position of object or None if not found
        
        Tries both approaches:
        1. Direct simulator query (fast, reliable)
        2. Vision-based detection (realistic, slower)
        """
        # Approach A: Direct simulator query
        pos = self.get_object_position(object_name)
        if pos is not None:
            return pos
        
        # Approach B: Vision-based fallback
        image = self.render_scene_image()
        detections = self.detect_color_objects(image)
        
        color = object_name.split("_")[0].lower()
        if color in detections:
            cx, cy = detections[color]
            world_pos = self.pixel_to_world(cx, cy)
            return world_pos
        
        return None
    
    def detect_all_objects(self):
        """
        Detect all objects in scene
        Returns dict of {object_name: (x, y, z)}
        """
        all_objects = {}
        for obj_name in self.object_map.keys():
            pos = self.detect_object(obj_name)
            if pos is not None:
                all_objects[obj_name] = pos
        return all_objects
