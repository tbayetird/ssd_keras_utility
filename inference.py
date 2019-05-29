from config import pirogues_mer,pirogues_plage
from ssd_inference import *

# inference_on_big_video(
#                         ssd300_pirogues_mer,
#                         'D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0.mp4',
#                         'full_stitched_video.mp4',
#                         2000,
#                         0.3
#                         )

# inference_on_folder(
#                     ssd300_pirogues_mer,
#                     'D:\\datas\\pirogues-mer\\',
#                     True,
#                     0.3
# )

truncate_inference_on_video(
                        ssd300_pirogues_mer,
                        'D:\\datas\\Kayar 28.05.2019\\DJI_0040.mp4',
                        change_save_dir=True,
                        tracking=True,
                        half_bottom_only=True,
                        truncate_low=2400,
                        truncate_up=2900
)
