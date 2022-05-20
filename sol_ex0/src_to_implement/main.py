import pattern
from pattern import Checker, Circle, Spectrum


checker =Checker(20,2)
checker.draw()
checker.show()

circle =Circle(1024, 200, (512, 256))
circle.draw()
circle.show()

spectrum = Spectrum(532)
spectrum.draw()
spectrum.show()


