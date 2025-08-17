# h261enc_cuda
A useless H.261 encoder that does everything except bitstream writing in CUDA on the GPU


I do not know if this even works anymore. It used to.

I just found it on some old backup of mine and given the lack of examples why video encoder should **NOT** be done on GPUs I thought it would be a good idea to publish it on github.

Small Update: I switched this to some simple cmake and did some changes so CUDA >= 12.x works. If you give it a try you might want to change the hardcoded CUDA architecture in the cmake file...