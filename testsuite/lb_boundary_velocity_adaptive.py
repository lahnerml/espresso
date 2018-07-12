from __future__ import print_function
import espressomd.lb
import espressomd.lbboundaries
import espressomd.p4est
import espressomd.shapes
import time
import unittest as ut


@ut.skipIf(not espressomd.has_features(["LB_ADAPTIVE"]) or
           not espressomd.has_features(["LB_BOUNDARIES"]),
           "Features not available, skipping test.")
class LBBoundaryVelocityTest(ut.TestCase):
    """Test slip velocity of boundaries.

       In this simple test add wall with a slip verlocity is
       added and checkeckt if the fluid obtains the same velocity.
    """

    side_length = 8.0
    system = espressomd.System(box_l=[side_length, side_length, side_length])
    n_nodes = system.cell_system.get_state()["n_nodes"]
    system.seed = range(n_nodes)

    system.time_step = 1.0
    system.cell_system.skin = 0.1

    def test(self):
        system = self.system

        p4est = espressomd.p4est.P4est(
            min_ref_level=2, max_ref_level=4)
        system.actors.add(p4est)
        lb_fluid = espressomd.lb.LBFluid(
            agrid=1.0, dens=1.0, visc=1.0, fric=1.0, tau=0.03)
        system.actors.add(lb_fluid)

        v_boundary = [0.03, 0.02, 0.01]

        wall_shape = espressomd.shapes.Wall(normal=[1, 2, 3], dist=1.0)
        wall = espressomd.lbboundaries.LBBoundary(shape=wall_shape,
                                                  velocity=v_boundary)
        system.lbboundaries.add(wall)
        p4est.random_refinement(2)
        # lb_fluid.print_vtk_boundary("test_bnd_0")
        p4est.regional_coarsening([1., 7., 1., 7., 1., 7.])
        # lb_fluid.print_vtk_boundary("test_bnd_1")
        p4est.regional_refinement([2., 6., 2., 6., 2., 6.])
        # lb_fluid.print_vtk_boundary("test_bnd_2")
        p4est.geometric_refinement()
        # lb_fluid.print_vtk_boundary("test_bnd_3")

        n_steps = 3072
        i_step = n_steps

        # lb_fluid.print_vtk_boundary("test_bnd")
        for i in range(int(n_steps / i_step)):
            system.integrator.run(i_step)
        # lb_fluid.print_vtk_velocity(path="tst_bnd_vel_{0:05d}".format(n_steps))

        v_fluid = lb_fluid[5, 0, 0].velocity
        self.assertAlmostEqual(v_fluid[0], v_boundary[0], places=3)
        self.assertAlmostEqual(v_fluid[1], v_boundary[1], places=3)
        self.assertAlmostEqual(v_fluid[2], v_boundary[2], places=3)


if __name__ == "__main__":
    ut.main()
