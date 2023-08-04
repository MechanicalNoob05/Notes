#include <stdlib.h>
#include <stdio.h>

int main()
{
    int n;
    int isPrime=0;
    printf("Enter a number: ");
    scanf("%d",&n);
    for (int i =2;i<=n/2 ;i++) {
        if (n%i==0) {
            isPrime=1;
            break;
        }
    }
    if(isPrime==0){
            printf("%d is a prime number\n",n);
    }else{
            printf("%d is a not prime number\n",n);
    }
    return 0;
}
