from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


class SimulationReport:
    """This class stores the simulation state over time. Useful for reporting.
    
    Instances should be periodically updated with the prevailing simulation
    state using update(). At the end of the simulation call finalise().
    
    Attributes:
        table: A dictionary that records the simulation state over time.
    """
    
    table: Dict[str, List[Any]] = {}
    
    def update(self, sim_state: Dict[str, Any]) -> None:
        """Append the newest simulation state to table and call _notify_update().
        
        Args:
            sim_state: A dictionary that characterises the simulation state. It
                should consist of the same keys every time. Intended to be the
                dict returned by TwoTempSolver.update(), but need not be.
        """
        for key, value in sim_state.items():
            if key not in self.table:
                self.table[key] = []
            self.table[key].append(value)
        self._notify_update()

    def _notify_update(self) -> None:
        """The table has been updated."""
        pass

    def finalise(self) -> None:
        """The simulation has ended."""
        pass

    def sum_columns(self, *keys: str) -> List[Any]:
        """Sum any number of columns from the table.
        
        Args:
            keys: Table keys of columns you wish to sum.
        
        Returns:
            The element-wise sum of table[key] over each key in keys.
        """
        sum_list = self.table.get(keys[0], []).copy()
        for key in keys[1:]:
            sum_list = [sum(values) for values in zip(sum_list,
                                                      self.table.get(key, []))]
        return sum_list


class SummaryVisualiser(SimulationReport):
    """At the end of the simulation, this report generates various graphs
    to depict the time evolution of the system.
    
    It is assumed that the following keys have been defined in the table:

        t, te, tn, ke, ee, w, ce, cce, tot, qe, cfg.
    
    Consult TwoTempSolver for their meaning.
    """
    
    def finalise(self) -> None:
        self._display_plots()

    def _display_plots(self) -> None:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs = axs.flatten()
        
        # Temperatures
        axs[0].plot(
            self.table['t'],
            self.table['te'],
            label='Electronic')
        axs[0].plot(
            self.table['t'],
            self.table['tn'],
            label='Nuclear')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Temperature')
        
        # Energies
        axs[1].fill_between(
            self.table['t'],
            0., self.sum_columns('ke', 'ee', 'w'),
            label='$W$')
        axs[1].fill_between(
            self.table['t'],
            0., self.sum_columns('ke', 'ee'),
            label='$E_e$')
        axs[1].fill_between(
            self.table['t'],
            0., self.table['ke'],
            label='KE')
        axs[1].fill_between(
            self.table['t'],
            self.table['u'], 0.,
            label='U')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Energy')

        # Heat capacities
        axs[2].fill_between(
            self.table['t'],
            0., self.table['ce'],
            alpha=0.5, label='$C_e$')
        axs[2].fill_between(
            self.table['t'],
            0., self.table['cce'],
            alpha=0.5, label='Corrective')
        axs[2].plot(
            self.table['t'],
            self.sum_columns('ce', 'cce'),
            c='k', label='Combined')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Electronic heat capacity')
        
        # Energy drift
        table_tot_rel = self.table['tot'] - self.table['tot'][0]
        axs[3].fill_between(
            self.table['t'],
            0., self.table['qe'],
            alpha=0.5, label='Absorbed energy')
        axs[3].fill_between(
            self.table['t'],
            self.table['qe'], table_tot_rel,
            label='Drift')
        axs[3].plot(
            self.table['t'],
            table_tot_rel,
            color='k', label='Total energy')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Energy change')
        
        for ax in axs[:4]:
            ax.legend()

        # Initial configuration
        axs[4].remove()
        axs[4] = fig.add_subplot(2, 3, 5, projection='3d')
        axs[4].scatter(self.table['cfg'][0][:,0],
                       self.table['cfg'][0][:,1],
                       self.table['cfg'][0][:,2])
        axs[4].set_title('Initial configuration')

        # Final configuration
        axs[5].remove()
        axs[5] = fig.add_subplot(2, 3, 6, projection='3d')
        axs[5].scatter(self.table['cfg'][-1][:,0],
                       self.table['cfg'][-1][:,1],
                       self.table['cfg'][-1][:,2])
        axs[5].set_title('Final configuration')
        
        plt.tight_layout()
        plt.show()
