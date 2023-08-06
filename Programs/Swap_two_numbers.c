#include <stdio.h>

int main(){
    int first,second,temp;
    printf("Enter first Number\n");
    scanf("%d",&first);
    printf("Enter second Number\n");
    scanf("%d",&second);

    printf("Numbers before swap %d %d\n",first,second);

    temp=first;
    first=second;
    second=temp;
    printf("Numbers after swap %d %d\n",first,second);

    return 0;
}
