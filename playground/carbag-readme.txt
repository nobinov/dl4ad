properties of the column:
imgseq : number of image sequence, correspond to image file name
timestamp : timestamp from ros
broken : flag to mark which image that have no pose (pose broken due to undetected) - True(have no pose)
x,y,z,w : pose of the car. all 0 for broken image
