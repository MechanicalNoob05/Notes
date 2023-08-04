
    

#include <stdio.h>

int main()
{
    //Create a table of given number till specified value
    
    int num,till;
    printf("Give a value to create a table: ");
    scanf("%d",&num);
    printf("Give a stop value: ");
    scanf("%d",&till);
    for(int i=0;i<till;i++){
       printf("%d x %d = %d\n",num,i+1,num*(i+1));
    };
    
    return 0;
}
