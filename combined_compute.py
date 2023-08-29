from compute_msd_jump import main as msd_jump_main
from compute_msd_sine import main as msd_sine_main
from compute_msd_quad_state_comp import main as state_comp_main
from compute_phdmd_init_diff import main as init_comp_main
from compute_wave_jump import main as wave_jump_main
from compute_wave_sine import main as wave_sine_main


if __name__ == '__main__':
    runs = [
        ('MSD Jump', msd_jump_main),
        ('MSD Sine', msd_sine_main),
        ('MSD State Comp', state_comp_main),
        ('MSD Init Comp', init_comp_main),
        ('Wave Jump', wave_jump_main),
        ('Wave Sine', wave_sine_main),
    ]

    for name, fun in runs:
        print(f'Running {name}...')
        fun()
