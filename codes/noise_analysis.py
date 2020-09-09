import glob
import os
import parselmouth
import numpy as np

def init_set_plt():
  import matplotlib.pyplot as plt
  # Initial plt settings
  plt.rcParams["figure.figsize"] = (8, 6)
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams['axes.titlepad'] = 10
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.serif'] = ['Times New Roman']
  return plt

def draw_pitch(filename, output_fn=None):
  plt = init_set_plt()
  snd = parselmouth.Sound(filename)
  pitch = snd.to_pitch()
  pitch_values = pitch.selected_array['frequency']
  proportion = len(pitch_values[pitch_values > 0]) / len(pitch_values)

  print("="*80)
  print(f"Filename: {filename}")
  print(f"Voiced segment proportion: {proportion}")
  print("="*80 + "\n")

  pitch_values[pitch_values==0] = np.nan
  plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
  plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
  plt.grid(False)
  plt.xlim([snd.xmin, snd.xmax])
  plt.ylim(50, 450)
  plt.xlabel("Time (s)", fontsize=24)
  plt.ylabel("Pitch (Hz)", fontsize=24)
  plt.title(filename.split("-")[0], fontsize=20)
  plt.tight_layout()
  if output_fn is not None:
    plt.savefig(output_fn)
  plt.close()


if __name__ == "__main__":
  for filename in glob.glob(os.path.join("detect_noise", "*.wav")):
    output_fn = "{}.png".format(filename.split("-")[0])
    draw_pitch(filename, output_fn)