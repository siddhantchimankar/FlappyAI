import pygame
import random
import neat
import time
import os

pygame.font.init()

WIDTH = 480
HEIGHT = 720
loop = 0

BIRDIMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join(
    'imgs', 'bird1.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join(
        'imgs', 'bird2.png'))), pygame.transform.scale2x(
            pygame.image.load(os.path.join('imgs', 'bird3.png')))]

PIPIMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))

BASEIMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))

BGIMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))

STATFONT = pygame.font.SysFont('comicsans', 50)

class Bird:
    IMGS = BIRDIMGS
    MAXROTATION = 25
    ROTVEL = 20
    ANIMETIME = 5

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.tilt = 0
        self.tickcount = 0
        self.vel = 0
        self.height = self.y
        self.imgcount = 0
        self.img = self.IMGS[0]


    def jump(self):
        self.vel = -10.5
        self.tickcount = 0
        self.height = self.y


    def move(self):
        self.tickcount += 1

        d = self.vel*self.tickcount + 1.5*self.tickcount**2

        if(d >= 16):
            d = 16

        if(d < 0):
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAXROTATION:
                self.tilt = self.MAXROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTVEL


    def draw(self, win):
        self.imgcount += 1

        if self.imgcount < self.ANIMETIME:
            self.img = self.IMGS[0]
        elif self.imgcount < self.ANIMETIME*2:
            self.img = self.IMGS[1]
        elif self.imgcount < self.ANIMETIME*3:
            self.img = self.IMGS[2]
        elif self.imgcount < self.ANIMETIME*4:
            self.img = self.IMGS[0]
            self.imgcount = 0
        # elif self.imgcount == self.ANIMETIME*4 + 1:
        #     self.img = self.IMGS[0]
        #     self.imgcount = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.imgcount = self.ANIMETIME*2

        rotatedimage = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotatedimage.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotatedimage, new_rect.topleft)
    


    def getmask(self):
        return pygame.mask.from_surface(self.img)




class Pipe:
    GAP = 170 
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPETOP = pygame.transform.flip(PIPIMG, False, True)
        self.PIPEBOTTOM = PIPIMG

        self.passed = False
        self.set_height()


    def set_height(self):
        self.height = random.randrange(50, 350)
        self.top = self.height - self.PIPETOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPETOP, (self.x, self.top))
        win.blit(self.PIPEBOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        birdmask = bird.getmask()
        topmask = pygame.mask.from_surface(self.PIPETOP)
        bottommask = pygame.mask.from_surface(self.PIPEBOTTOM)

        topoffset = (self.x - bird.x, self.top - round(bird.y))
        bottomoffset = (self.x - bird.x, self.bottom - round(bird.y))

        bpoint = birdmask.overlap(bottommask, bottomoffset)
        tpoint = birdmask.overlap(topmask, topoffset)

        if tpoint or bpoint:
            return True

        return False

    def get_mask(self):

        return pygame.mask.from_surface(self.img)




class Base:
    VEL = 5
    WIDTH = BASEIMG.get_width()
    IMG = BASEIMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))






def drawwin(win, birds, pipes, base, score):
    win.blit(BGIMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STATFONT.render("Score: " + str(score), 1,(255, 255, 255))
    win.blit(text, (WIDTH - 10 - text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    global loop
    loop += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        birds.append(Bird(230, 350))
        g.fitness = 0
        nets.append(net)
        ge.append(g)



    base = Base(640)
    pipes = [Pipe(500)]
    score = 0
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    if(loop == 1):
        pygame.time.delay(10000)
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()



        pipeind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPETOP.get_width():
                pipeind = 1
        else:
            run = False
            break


        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            # print(pipeind)
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipeind].height), abs(bird.y - pipes[pipeind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        # bird.move()
        add_pipe = False
        
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)


                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True



            if pipe.x + pipe.PIPETOP.get_width() < 0:
                rem.append(pipe)

            

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5

            pipes.append(Pipe(500))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() > 720 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        drawwin(win, birds, pipes, base, score)





def run(configpath):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configpath)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    configpath = os.path.join(local_dir, 'config.txt')
    run(configpath)

