from config import flowers,pirogues_mer,pirogues_plage
from ssd_inference import *

inference_on_big_video(
                        ssd300_pirogues_mer,
                        'D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0.mp4',
                        'full_stitched_video.mp4',
                        2000,
                        0.3
                        )

# inference_on_folder(
#                     ssd300_pirogues_mer,
#                     'D:\\datas\\pirogues-mer\\',
#                     True,
#                     0.3
# )
