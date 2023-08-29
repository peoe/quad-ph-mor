from evaluate_msd import main as msd_main
from evaluate_wave import main as wave_main
from filter_results import main as filter_main


if __name__ == '__main__':
    runs = [
        ('MSD Evaluation', msd_main),
        ('Wave1D Evaluation', wave_main),
        ('Filter', filter_main),
    ]

    for name, fun in runs:
        print(f'Running {name}...')
        fun()
