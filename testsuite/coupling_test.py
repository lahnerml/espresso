import sys
import unittest as ut
import numpy as np
import numpy.testing
import espressomd
from espressomd import lb
from itertools import product


@ut.skipIf(not espressomd.has_features("LB_ADAPTIVE") or
           not espressomd.has_features("LB"),
           "Features not available, skipping test!")
class TestLBGetUAtPos(ut.TestCase):
    """
    Check velocities at particle positions are sorted by ``id`` and
    quantitatively correct (only LB GPU).

    """
    @classmethod
    def setUpClass(self):
        self.params = {
            'tau': 0.01,
            'agrid': 1.0,
            'box_l': [2.0, 2.0, 2.0],
            'dens': 0.85,
            'viscosity': 30.0,
            'friction': 2.0,
            'gamma': 1.5
        }

        self.n_pos_to_check = 25
        self.n_probes = 256
        self.system = espressomd.System(box_l=[1.0, 1.0, 1.0])
        self.system.box_l = self.params['box_l']
        self.system.cell_system.skin = 0.2
        self.system.time_step = 0.01

        # setup dummy md grid
        self.system.non_bonded_inter[0, 0].lennard_jones.set_params(
            epsilon=1.0, sigma=1.0, cutoff=0.8, shift="auto")

        # setup fluid
        self.lb_fluid = lb.LBFluid(
            visc=self.params['viscosity'],
            dens=self.params['dens'],
            agrid=self.params['agrid'],
            tau=self.params['tau'],
            fric=self.params['friction']
        )
        self.system.actors.add(self.lb_fluid)

        # setup fluid velocity
        for n in product(range(2), range(2), range(1)):
            self.lb_fluid[n].velocity = [0., 0., 0.]
        for n in product(range(2), range(2), range(1, 2)):
            self.lb_fluid[n].velocity = [0., 0., 1.]

        # calculate positions (as well as expected velocity)
        self.pos = np.linspace(0, 1, num=self.n_pos_to_check, dtype=float)

    def test_get_u_at_pos(self):
        """
        Test if linear interpolated velocities are equal to the velocities at
        the particle positions. This test uses the two-point coupling under
        the hood.

        """
        for i in range(len(self.pos)):
            for j in range(self.n_probes):
                self.system.part.clear()
                self.system.part.add(
                        id=0,
                        pos=(np.random.random_sample(),
                             np.random.random_sample(),
                             0.5+self.pos[i]))
                self.system.integrator.run(0)
                v = self.lb_fluid.get_interpolated_velocity(
                        self.system.part[0].pos)
                np.testing.assert_allclose(v, [0., 0., self.pos[i]], 1e-10)
                # self.assertTrue(v[0] == 0. and
                #                 v[1] == 0. and
                #                 abs(v[2] - self.pos[i]) < 1e-10)


if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTests(ut.TestLoader().loadTestsFromTestCase(TestLBGetUAtPos))
    result = ut.TextTestRunner(verbosity=4).run(suite)
    sys.exit(not result.wasSuccessful())
