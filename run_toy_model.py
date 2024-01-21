"""Example of how to use the 2T-MD package.

This script creates a spherical nanoparticle with internal interactions defined
by ToyMaterial (this material has been contrived to produce a negative heat
capacity at high electronic temperatures). After initial thermal equilibration
the nanoparticle is irradiated with an ultra-fast laser pulse. A positive heat
capacity is maintained using an energy corrective.

Note that temperatures are expressed in energy units so that the package is free
from physical constants. It therefore works for any units you wish to use.
"""
import sys
import time

from corrective import DarkinsCorrective
from material import ToyMaterial
import radiation_source as radiation
import simulation_report as report
from two_temp_solver import TwoTempSolver
from two_temp_system import TwoTempNanoparticle


def main(unused_argv):
    initial_temp = 0.01
    material = ToyMaterial()
    ttsys = TwoTempNanoparticle(3., temp=initial_temp)
    dt = 1e-3 
    solver = TwoTempSolver(ttsys, material)
    
    # Equilibration
    solver.update(10., dt, initial_temp)
        
    # Production
    time_of_pulse = 0.
    solver.radiation = radiation.Laser(time_of_pulse, 1., 5.)
    solver.corrective = DarkinsCorrective(0.1, 0.02)
    solver.use_trotter = False
    solver.time = time_of_pulse - 5.
    visual = report.SummaryVisualiser()
    for i in range(150):
        state = solver.update(0.1, dt)
        # Record state
        visual.update(state)
        with open('trajectory.xyz', 'a') as xyz_file:
            ttsys.dump_xyz(xyz_file)
            
    visual.finalise()

if __name__ == '__main__':
    main(sys.argv)

