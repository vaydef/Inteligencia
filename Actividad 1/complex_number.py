class ComplexNumber(object):
    def __init__(self, re, im):
        self.re = re
        self.im = im

    def __add__(self, other):
        real = self.re + other.re
        imag = self.im + other.im
        return ComplexNumber(real,imag)

    def __sub__(self, other):
        real = self.re - other.re
        imag = self.im - other.im
        return ComplexNumber(real,imag)

    def __mul__(self, other):
        real = (self.re * other.re) - (self.im * other.im)
        imag = (self.re * other.im) + (self.im * other.re)
        return ComplexNumber(real,imag)

    def __truediv__(self, other):
        conjother = ComplexNumber(other.re,-1*other.im)
        numerador = self.__mul__(conjother)
        denominador = (other.re**2) + (other.im**2)
        real = numerador.re/denominador
        imag = numerador.im/denominador
        return ComplexNumber(real,imag)
        

    def __invert__(self):
        sq = (self.re**2) + (self.im**2)
        real = self.re/sq
        imag = -1*(self.im/sq)
        return ComplexNumber(real,imag)
        
    def __abs__(self):
        return ((self.re**2) + (self.im**2))**(1/2)

    def __eq__(self, other):
        if self.re == other.re and self.im == other.im:
            return True
        return False
            

    def __repr__(self):
        if self.im<0:
            r = str(round(self.re,2))+" - "+str(-1*round(self.im,2))+"i"
        else:
            r = "{0:.2f}".format(self.re)+" + "+"{0:.2f}".format(self.im)+"i"
        return r
            
