import cProfile, pstats, io
from pstats import SortKey
from cv2 import imread
import sys
sys.path.append('..')
from facematch import Facematch

img_1 = imread("./imagens_test/biden.png")
img_2 = imread("./imagens_test/licensed-image.jpeg")

result = Facematch.verify(img_1, img_2)

pr = cProfile.Profile()
pr.enable()

result = Facematch.verify(img_1, img_2)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

benchmark = s.getvalue()

file = open("benchmark.txt", "w")
file.write(benchmark)
file.close()

