import torch
from botorch.utils.sampling import draw_sobol_samples
import pandas as pd
import numpy as np

def round_to_significant_figures(num, n=2):
    if num == 0:
        return 0.0
    return round(num, n - 1 - int(np.floor(np.log10(abs(num)))))

def botorch_sobol_sorted(param_bounds, n_samples, seed=42):
    param_names = list(param_bounds.keys())
    dim = len(param_names)
    bounds_norm = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float)

    torch.manual_seed(seed)
    np.random.seed(seed)

    normalized_samples = draw_sobol_samples(bounds=bounds_norm, n=n_samples, q=1).squeeze(1)
    actual_samples_rounded = []
    rounded_normalized = torch.zeros_like(normalized_samples)

    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        norm_val = normalized_samples[:, i]

        if name == 'oxygen_pressure':
            log_val = low + norm_val * (high - low)
            actual_val = torch.pow(10, log_val)
        else:
            actual_val = low + norm_val * (high - low)

        if name == 'frequency':
            rounded_actual = torch.round(actual_val)
        elif name == 'temperature':
            rounded_actual = torch.round(actual_val * 10) / 10.0
        else:
            rounded_actual = torch.round(actual_val * 100) / 100.0

        if name == 'oxygen_pressure':
            rounded_actual = torch.tensor([round_to_significant_figures(x.item(), 3) for x in actual_val])

        actual_samples_rounded.append(rounded_actual.numpy())

        if name == 'oxygen_pressure':
            log_rounded = torch.log10(rounded_actual)
            rounded_norm = (log_rounded - low) / (high - low)
        else:
            rounded_norm = (rounded_actual - low) / (high - low)

        rounded_normalized[:, i] = rounded_norm

    distances = torch.norm(rounded_normalized - 0.5, dim=1).numpy()
    df = pd.DataFrame(np.column_stack(actual_samples_rounded), columns=param_names)
    df['normalized_distance'] = distances
    df.sort_values(by='normalized_distance', inplace=True)

    return df

# Parameter bounds
param_bounds = {
    'oxygen_pressure': (-4, 0),
    'laser_energy_density': (1, 3),
    'temperature': (500, 800),
    'frequency': (1, 10),
    'thickness': (5, 200),
}

n_samples = 32  # 2^5=32
min_distance = float('inf')
best_seed = None
best_df = None
valid_seeds = []

for seed in range(1, 101):  # Test seeds 1-100
    try:
        df = botorch_sobol_sorted(param_bounds, n_samples, seed)
        top_16 = df.head(16)

        freq_values = top_16['frequency'].astype(int).unique()

        if set(range(1, 10)).issubset(set(freq_values)):
            valid_seeds.append(seed)
            distance_16th = top_16.iloc[15]['normalized_distance']

            if distance_16th < min_distance:
                min_distance = distance_16th
                best_seed = seed
                best_df = df.copy()
                print(f"Valid seed {seed}: 16th sample distance = {distance_16th:.6f} (current best)")

    except Exception as e:
        print(f"Seed {seed} failed: {str(e)}")
        continue

if best_seed is not None:
    print(f"\nOptimal seed found: {best_seed}")
    print(f"Number of valid seeds: {len(valid_seeds)}")
    print(f"16th sample distance: {min_distance:.6f}")

    display_df = best_df.head(32).copy()

    def format_oxygen_pressure(x):
        return f"{x:.2e}"

    display_df['oxygen_pressure'] = display_df['oxygen_pressure'].apply(format_oxygen_pressure)
    display_df['laser_energy_density'] = display_df['laser_energy_density'].apply(lambda x: f"{x:.2f}")
    display_df['thickness'] = display_df['thickness'].apply(lambda x: f"{x:.2f}")
    display_df['temperature'] = display_df['temperature'].apply(lambda x: f"{x:.1f}")
    display_df['frequency'] = display_df['frequency'].astype(int)
    display_df['normalized_distance'] = display_df['normalized_distance'].apply(lambda x: f"{x:.6f}")

    top_16_freq = display_df.head(16)['frequency'].unique()
    print(f"\nFrequency values in top 16 samples: {sorted(top_16_freq)}")

    display_df.to_csv('optimal_sobol_samples.csv', index=False)
    print("\nTop 16 samples:")
    print(display_df.head(16))

    print("\nSample oxygen_pressure values:")
    print(display_df['oxygen_pressure'].head())
else:
    print("\nNo seed found with all frequencies 1-9 in top 16 samples")
    if valid_seeds:
        print(f"Found {len(valid_seeds)} seeds with partial coverage")