from __future__ import absolute_import, print_function, division, unicode_literals
import time
import math
import sys
import os, errno


def makedirs(dirname):
    """Creates the directories for dirname via os.makedirs, but does not raise
       an exception if the directory already exists and passes if dirname=""."""
    if not dirname:
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def progressinfo(seq, verbose=True, length=None, prefix=''):
	"""Yields from seq while displaying progress information.
	   Unlike mdp.utils.progessinfo, this routine does not
	   display the progress after each iteration but tries
	   to approximate adaequate stepsizes in order to print
	   the progress information roughly ones per second.

       -verbose: if False, the function behaves like `yield from seq`
       -length: can be used to give the length of sequences that
             have no __len__ attribute or to overwrite the length
       -prefix: Will be printed before the status information.
	"""
	if not verbose:
		for item in seq:
			yield item
		return
	next_step = 1
	step_size = 1
	last_time = time.time()
	start_time = last_time
	if length is None:
		if hasattr(seq, '__len__'):
			length = len(seq)
	if length is not None:
		prec = int(math.ceil(math.log10(length)))
		out_string = "\r{prefix}{{count:{prec}d}} ({{ratio:3.1f}}%)".format(prec=prec, prefix=prefix)
	else:
		length = 1
		out_string = "\r{prefix}{{count:d}}".format(prefix=prefix)
	steps = 0
	for i, item in enumerate(seq):
		yield item
		if i == next_step:
			this_time = time.time()
			time_diff = this_time - last_time + 0.0001
			normed_timediff = time_diff / (step_size)
			new_step_size = int(math.ceil(1.0/normed_timediff))
			#In order to avoid overshooting the right stepsize, we take
			#a convex combination with the old stepsize (that will be
			#too small at the beginning)
			step_size = int(math.ceil(0.8*step_size+0.2*new_step_size))
			last_time = this_time
			next_step = i+step_size
			print(out_string.format(count=i, ratio=1.0*i/length*100), end='')
			sys.stdout.flush()
			#steps += 1
	print(out_string.format(count=length, ratio=100.0))
	#end_time = time.time()
	#print "Needed Steps: ", steps
	#print "Last stepsize", step_size
	#print "Needed time", end_time - start_time
	#print "Avg time per step", (end_time - start_time) / steps

def getChunks(seq,verbose=True):
	"""Yields chunks from seq while optionally displaying progress information.
	   after each chunk.
	   This routine tries
	   to approximate adaequate chunksizes in order to print
	   the progress information roughly ones per second.
	"""
	next_step = 1
	step_size = 1
	last_time = time.time()
	start_time = last_time
	length = len(seq)
	prec = int(math.ceil(math.log10(length)))
	out_string = "\r %{0}d (%3.1f %%)".format(prec)
	steps = 0
	next_chunk = []
	for i, item in enumerate(seq):
		next_chunk.append(item)
		if i == next_step:
			yield next_chunk
			next_chunk = []
			this_time = time.time()
			time_diff = this_time - last_time + 0.0001
			normed_timediff = time_diff / (step_size)
			new_step_size = int(math.ceil(1.0/normed_timediff))
			#In order to avoid overshooting the right stepsize, we take
			#a convex combination with the old stepsize (that will be
			#too small at the beginning)
			step_size = int(math.ceil(0.8*step_size+0.2*new_step_size))
			last_time = this_time
			next_step = i+step_size
			if verbose:
				print(out_string % (i, 1.0*i/length*100), end='')
			sys.stdout.flush()
			#steps += 1
	if next_chunk:
		yield next_chunk
	print(out_string % (length, 100.0))
	#end_time = time.time()
	#print "Needed Steps: ", steps
	#print "Last stepsize", step_size
	#print "Needed time", end_time - start_time
	#print "Avg time per step", (end_time - start_time) / steps

def arange_list(l, maxcols=None, empty=False):
    pass

if __name__ == '__main__':
	#new_list= []
	#for chunk in getChunks(range(10000), prefix='test'):
	#	new_list.extend(chunk)
	#assert(new_list == range(10000))

	for i in progressinfo(range(1000), prefix='test'):
		pass
