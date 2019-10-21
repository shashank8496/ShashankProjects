#include<reg52.h>

sbit sw=P3^0; //switch for interrupt

int b=0,z=0,l=0,c=0;

int d=0;

int x=0;

int e=0;//temporary variable for dice value

void delay(unsigned int);

void setled(int);

void rowset(int);

void intpt(int);

void main()

{

unsigned char s[6]={0x7D,0x6D,0x66,0x5B,0x06,0x4F,};

unsigned int z[6]={6,5,4,2,1,3};

while(1)

{

while(sw==1)

{

for(l=0;l<6;l++) //dice value generator code

{

P0=s[l];

x=z[l];

delay(5);

if(sw==0)

{

P0=s[l];

intpt(x);

x=0;

delay(50);

}

}//for end

l=0;

}//while(sw) end

}//while(1) end

} // main end

void intpt(int z)//interrupt function

{

c=z;

e=d;

d=d+c;

if((d>=1)&&(d<=48))

{

if(d==5) //for ladder

{

d=d+9;

}

if(d==29) //for ladder

{

d=d+8;

}

else if(d==39) //for snakes

{

d=d-17;

}

else if(d==31)

{

d=d-11;

}

else if(d==47)

{

d=d-15;

}


setled(d);

}

else // if value exceeds 48

{

d=e;

setled(d);

}

} //end of interrupt

void delay(unsigned int times) //delay function

{

unsigned int k,j;

k=0;

j=0;

for(k=0;k<times;k++)

{

for(j=0;j<1275;j++);

}

} //end of delay function

void setled(i) //interfacing seven segment led

{

if((i==1)||(i==16)||(i==17)||(i==32)||(i==33)||(i==48))

{

P1=0x01;

rowset(i);

delay(10);

}

else if((i==2)||(i==15)||(i==18)||(i==31)||(i==34)||(i==47))

{

P1=0x02;

rowset(i);

delay(10);

}

else if((i==3)||(i==14)||(i==19)||(i==30)||(i==35)||(i==46))

{

P1=0x04;

rowset(i);

delay(10);

}

else if((i==4)||(i==13)||(i==20)||(i==29)||(i==36)||(i==45))

{

P1=0x08;

rowset(i);

delay(10);

}

else if((i==5)||(i==12)||(i==21)||(i==28)||(i==37)||(i==44))

{

P1=0x10;

rowset(i);

delay(10);

}

else if((i==6)||(i==11)||(i==22)||(i==27)||(i==38)||(i==43))

{

P1=0x20;

rowset(i);

delay(10);

}

else if((i==7)||(i==10)||(i==23)||(i==26)||(i==39)||(i==42))

{

P1=0x40;

rowset(i);

delay(10);

}

else if((i==8)||(i==9)||(i==24)||(i==25)||(i==40)||(i==41))

{

P1=0x80;

rowset(i);

delay(10);

}

else

{

P1=0x00;

P2=0x00;

delay(10);

}

}

void rowset(i)

{

if((i>=1)&&(i<=8))

{

P2=0xFE;

delay(10);

}

else if((i>=9)&&(i<=16))

{

P2=0xFD;

delay(10);

}

else if((i>=17)&&(i<=24))

{

P2=0xFB;

delay(10);

}

else if((i>=25)&&(i<=32))

{

P2=0xF7;

delay(10);

}

else if((i>=33)&&(i<=40))

{

P2=0xEF;

delay(10);

}

else if((i>=41)&&(i<=48))

{

P2=0xDF;

delay(10);

}

delay(100);

}