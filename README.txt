what if you used a 3d unet diffusion model to generate minecraft chunks?
its horribly inefficient but kinda interesting!

##
#  setup (train on own world):
##

- place a minecraft world in training-world
  (such that training-world/level.dat exists)

- run ./train.sh to train a model on that world. it should have about 20k samples

- wait forever. it took 24 hours on a 3090 to reach epoch 46 for me.

- run ./test.sh to generate some test chunks. images will be placed in ./tests/

##
#  setup (use pretrained latest.pt)
##

- just run ./test.sh