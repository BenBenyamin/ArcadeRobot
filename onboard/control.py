
from stable_baselines3.common.base_class import BaseAlgorithm
import stretch_body.robot
from torch import inference_mode

class StretchControl:

    def __init__(self, algorithm:BaseAlgorithm, model_path:str,action_remapping:dict):

        """
        action remapping = {
        0 : 0.03, #NO OP
        1 : 0.04 # UP
        2 : 0.02 # DOWN
        }
        """
        self.model = algorithm.load(path=model_path)
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        self.vel = self.robot.arm.params['motion']['max']['vel_m']
        self.acc = self.robot.arm.params['motion']['max']['accel_m']
        
        print(f"\nArm velocity: {self.vel} m/s\nArm acceleration: {self.acc} m/s²")

        self.action_remapping = action_remapping

        self.target_pos = self.action_remapping[0] ## target NO-OP

        self.move_arm(self.target_pos)

    def move_arm(self,x_m):

        self.target_pos = x_m
        self.robot.arm.move_to(x_m,v_m=self.vel, a_m=self.acc)
        self.robot.push_command()
    
    def ready(self):

        return abs(self.robot.arm.status['pos'] - self.target_pos) <= 0.001

    @inference_mode
    def act(self,obs):

        action, _ = self.model.predict(obs, deterministic=True)

        action = int(action)

        # convert it to UP DOWN NOOP
        self.move_arm(self.action_remapping[action])
    
    def command(self,obs):

        if not self.ready(): return

        self.act(obs)



        
