#include <stdio.h>

int main()
{

   // Sum of all natural number
   int a,b;
   printf("Sum till what number: ");
   scanf("%d",&a);

   // To reduce time complexity for loop is avoided 
   //for(int i = 1 ; i <= a; i++){
   //    b+=i;
   //}

   // the essence of belove formula is 
   // S = 1+  2  +  3  +  4  +  5  ....+n can also be written as
   // S = n+(n+1)+(n+2)+(n+3)+(n+4)....+1
   // adding both we get 
   // 2S = (n+1)+(n+1)+(n+1)+(n+1)....+(n+1)
   // S = n(n+1)/2
   // i.e S = (n*n+n)/2

   b=(a*a+a)/2;
   printf("%d\n",b);

   return 0;
}
