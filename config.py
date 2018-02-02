class OrdinaryWasserSteinShallow:
  param = {
    "learning_rate" : 0.00005   ,   #0.001 default rmsprop. Lower for wasserstein
    "batch_size"    : 20        ,

    "start"         : 1         ,
    "end"           : 10001     ,
    "checkpoint"    : 300       ,
    "_embedding"    : None      ,
    "testpoint"     : 100       ,
    "time_steps"    : 45        ,
    "vector_size"   : 200       ,

    #wasserstein parameters
    "n_critic"      : 4         ,   #number of times to train discriminator for every train of gen. Improves gradient
    "w_clipping"    : 0.01      ,   #-w_clipping < w < w_clipping for every weight in discriminator. theorical requirement

    #dropout parameters
    "drop_D_inter"  : 0.1       ,   #in between convolutional discriminator layers.
    "drop_input"    : 0.2       ,   #dropout on input data
    "drop_output"   : 0.5       ,   #dropout on output
    "drop_rnn"      : 0.25      ,   #dropout in recurrent connections
  }

# class otherNetwork:
#     param={}
#
