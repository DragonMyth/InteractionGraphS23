import numpy as np

from mocap_processing.utils import conversions
from mocap_processing.utils import conversions_scipy

from mocap_processing.viz.utils import TimeChecker

print('-------- Conversion Test --------')

scale = 1000

modules = [conversions, conversions_scipy]

As = np.random.uniform(-2.0, 2.0, size=(scale, 3))
Rs = conversions.A2R(As)
Qs = conversions.A2Q(As)

tests = ['A2R', 'A2Q', 'R2A', 'R2E', 'R2Q', 'Q2R', 'Q2A', 'Q2E']
results = [list() for i in range(len(modules))]

tc = TimeChecker()

for i, m in enumerate(modules):
	res = results[i]

	''' 
	Check whether the module generates axis-angle 
	whose angle is larger than PI
	'''
	print('Gen Axis-angle (angle>pi)?: ', 
		np.any(np.linalg.norm(m.R2A(Rs), axis=1) > np.pi))

	''' A2R '''
	tc.begin()
	m.A2R(As)
	res.append(tc.get_time())

	''' A2Q '''
	tc.begin()
	m.A2Q(As)
	res.append(tc.get_time())

	''' R2A '''
	tc.begin()
	m.R2A(Rs)
	res.append(tc.get_time())

	''' R2E '''
	tc.begin()
	m.R2E(Rs)
	res.append(tc.get_time())

	''' R2Q '''
	tc.begin()
	m.R2Q(Rs)
	res.append(tc.get_time())

	''' Q2R '''
	tc.begin()
	m.Q2R(Qs)
	res.append(tc.get_time())

	''' Q2A '''
	tc.begin()
	m.Q2A(Qs)
	res.append(tc.get_time())

	''' Q2E '''
	tc.begin()
	m.Q2E(Qs)
	res.append(tc.get_time())

results = np.array(results).transpose()
judge = results[:, 0]/results[:, 1]

for i in range(len(tests)):
	print(tests[i], judge[i], results[i, :])

print('---------------------------------')