import sys, random
import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util

class animal:
    def __init__(self, space, color):
        self.mass = 0.1
        self.radius = 10
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.body = pymunk.Body(self.mass, self.inertia)
        x = random.randint(100,500)
        y = random.randint(100,500)
        self.setPosition(x,y)
        shape = pymunk.Circle(self.body, self.radius, (0,0))
        shape.color = color
        space.add(self.body, shape)
        return
    
    def setPosition(self,x,y):
        self.body.position = x, y
        return
    
    def setVelocity(self,vx,vy):
        self.body.velocity = vx, vy
        return

    def setForce(self,Fx,Fy):
        self.body.apply_force_at_local_point((Fx,Fy), (0,0))
        return

    def drag(self):
        dragconstant = 0.01
        vx, vy = self.body.velocity
        speed = (vx*vx + vy*vy)**0.5
        self.setForce(-vx*speed*dragconstant,-vy*speed*dragconstant)

    def act(self,action):
        self.body.apply_force_at_local_point((10*action[0],10*action[1]), (0,0))
        return


class Env(object):
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        self.dt = 1.0/10.0
        self.physicsPerFrame = 1
        self.numStep = 0
        
        # pygame
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Static barrier walls (lines) that the balls bounce off of
        self.add_wall()



        self.prey = animal(self.space,(255,0,0,255))
        self.predator = animal(self.space,(0,0,255,255))
        self.reset()


    def reset(self): 
        dist = 0
        while not(100 < dist < 300):
            x1 = random.randint(100,500)
            y1 = random.randint(100,500)
            x2 = random.randint(100,500)
            y2 = random.randint(100,500)
            dist = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5

        self.initDist = dist

        self.prey.setPosition(x1,y1)
        self.prey.setVelocity(0,0)
        self.predator.setPosition(x2,y2)
        self.predator.setVelocity(0,0)
        self.numStep = 0 
        return self.getPreyState(), self.getPredState()
    
    def getDist(self):
        x1, y1 = self.prey.body.position
        x2, y2 = self.predator.body.position
        dist = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5
        return dist

    def getPreyState(self):
        x1, y1 = self.prey.body.position
        vx1, vy1 = self.prey.body.velocity
        x2, y2 = self.predator.body.position
        vx2, vy2 = self.predator.body.velocity
        return [x1, y1, vx1, vy1, x2-x1, y2-y1, vx2, vy2]
        
    def getPredState(self):
        x1, y1 = self.predator.body.position
        vx1, vy1 = self.predator.body.velocity
        x2, y2 = self.prey.body.position
        vx2, vy2 = self.prey.body.velocity
        #return [x1, y1, vx1, vy1, x2-x1, y2-y1, vx2, vy2]  
        return [x2-x1, y2-y1, vx1, vy1]
    
    def getPreyReward(self):
        return 10
    
    def getPredReward(self):
        dist = self.getDist()
        if dist < self.prey.radius*2:
            return 100
        else :
            return -1

    def add_wall(self):

        static_body = self.space.static_body
        static_lines = [pymunk.Segment(static_body, (50.0, 50.0), (50.0, 550.0), 0.0),
                        pymunk.Segment(static_body, (50.0, 550.0), (550.0, 550.0), 0.0),
                        pymunk.Segment(static_body, (550.0, 550.0), (550.0, 50.0), 0.0),
                        pymunk.Segment(static_body, (550.0, 50.0), (50.0, 50.0), 0.0),]
        for line in static_lines:
            line.elasticity = 0.99
            line.friction = 0.9
        self.space.add(static_lines)
    


    def step(self,preyAct,predAct,renderOption):
        # do action
        self.prey.act(preyAct)
        self.predator.act(predAct)
        self.prey.drag()
        self.predator.drag()

        # step
        self.space.step(self.dt)
        self.numStep += 1

        # get new state 
        preyState = self.getPreyState()
        predState = self.getPredState()

        # get reward 
        preyReward = self.getPreyReward()
        predReward = self.getPredReward()
        
        # check done
        done = False
        if predReward >= 99 or self.numStep > 100:
            done = True

        #render
        if renderOption :

            self.screen.fill(THECOLORS["white"])
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(5000)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()) + "reward" +str(int(predReward*100)))


        # return state, reward, done 
        return preyState, preyReward, predState, predReward, done



