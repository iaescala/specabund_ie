c     @author: Ivanna Escala
c     smooth_gauss.f is a Fortran implementation of smooth_gauss.c, written by E.Kirby

      subroutine smooth_gauss (lambda1, spec1, lambda2, spec2, dlam,
     $     f, n1, n2, halfwindow)

      implicit real*8 (a-h,o-z)  
      integer i, j, n1, n2, low, high
      integer*4 f(n2), halfwindow
      real*8 lambda1(n1), spec1(n1), lambda2(n2), spec2(n2), dlam(n2),
     $     temp2, temp3, temp4, gauss
     
Cf2py intent(in) lambda1, spec1, lambda2, dlam, f, n1, n2, halfwindow
Cf2py intent(out) spec2

      do 10 i=1,n2

         low = f(i)-halfwindow
         if (low.lt.1) low = 1
         high = f(i)+halfwindow
         if (high.lt.1) high = 1
         if (high.gt.(n1)) high = n1

         if (low.lt.n1.and.low.lt.high) then

            temp2 = 0.
            temp3 = 0.
            temp4 = 0.

            do 20 j=low,high

               gauss = 0.
               if (abs(lambda1(j)-lambda2(i)).lt.dlam(i)*40.) then
                  gauss = exp(-1.0*(lambda1(j)-lambda2(i))*
     $                 (lambda1(j)-lambda2(i)) / (2.0*dlam(i)*dlam(i)))
     
                  temp2 = temp2 + gauss
                  temp3 = temp3 + spec1(j)*gauss
                  temp4 = temp4 + gauss*gauss

               endif
 20         continue
 
            if (temp2.gt.0) then
               spec2(i) = temp3 / temp2
               
            endif
         endif
 10   continue

      return
      end
