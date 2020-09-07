%macro p(mean1=,sq1=,mean2=,sq2=);
data step4;
run;
data b;
run;
%do l= 1 %to 50;
      %do k = 1 %to 20;
      data a;
            do i=1 to 6;
                  x = &mean1.+&sq1.*normal(ranuni(0));
                  y = &mean2.+&sq2.*normal(ranuni(0));
                  output;
            end;
      run;
      data b;
            set a b;
      run;
      %end;
      data c;
            set b;
            if x = '' then delete;
      run;
      proc plan seed=0;
            factors study=20 ordered
            id=6;
            output out=resultds;
      run;
      data long;
            merge c resultds;
            drop id;                    /*为什么drop id 而不是drop i*/
      run;
      proc sql;
            create table m as
            select mean(x) as x1,
                        mean(y) as x2
            from long
            group by study;
      run;
      quit;
      proc sql;
            create table n as
            select var(x) as S1,
                         var(y) as S2
            from long
            group by study;
      run;
      quit;
      data step1;
            merge m n;
            SP2=(S1+S2)/2;
            SS2=(1/3)*SP2;
            Y=X1-X2;
            W=1/SS2;
      RUN;
      PROC SQL;
            CREATE TABLE STEP2 AS
            SELECT SUM(W*Y*Y)-(SUM(W*Y)*SUM(W*Y))/SUM(W) AS Q
            FROM STEP1;
      RUN;
      QUIT;
      PROC SQL;
            CREATE TABLE STEP3 AS SELECT Q/19 AS H2
            FROM STEP2;
      RUN;
      QUIT;
      data u;
            merge step2 step3;
      run;
      data step4;
            set step4 u;
            if q = '' then delete;
      run;
%end;
%mend;


 %p(mean1=160,sq1=60,mean2=161,sq2=59);


data fan;
set step4;
if q<27.20 then count=0;
 else count=1;
run;

proc means data=fan n sum;
var count;
run;
/*        */
/*   */
/*       */
proc logistic data=fan;
  model count=q;
  score data=fan outroc=fan_roc;   
run;

data fan_p;
  set fan;
  logit=137.8-5.0539*q;           
  if logit<27.20 then count_predict=0;
  else count_predict=1;
  keep count logit count_predict;
run;
/*在输出可以看到一个四格表*/
proc freq data=fan_p;
  table count*count_predict;
run;                              

/*画roc曲线*/
axis order=(0 to 1 by .1) label=none length=4in;
symbol i=join v=none c=black;
symbol2 i=join v=none c=black;
proc gplot data = fan_roc;
plot _SENSIT_*_1MSPEC_ _1MSPEC_*_1MSPEC_/ overlay vaxis=axis haxis=axis;
run; 
quit;
/*这是第二种画roc的方法*/
ods graphics on;
proc logistic data=fan plots(only)=roc;
model count=q;
run;
ods graphics off;
