clear;clc;clf;%close all;

% startovaci reseni
M1_area = 18; M1_typ = 4; M1_tlak_perm = 5;
M2_area = 4; M2_typ = 4; M2_tlak_perm = 38.22;
M3_area = 4; M3_typ = 4;
C1_tlak = 318.515;
C2_tlak = 146.764;

% x0 = [M1_area, M1_typ, M1_tlak_perm, M2_area, M2_typ, M2_tlak_perm, M3_area, M3_typ, C1_tlak, C2_tlak];
x0 = [2.0,2.3849,69.4278,2.0,2.1571,74.5455,2.0,2.0831,259.2578,110.1235];


% referencni hodnoty pro puvodni reseni a ukazani hodnot toku a vseho mozneho 
% [ref, toky] = simulator(M1_area,M1_typ, M1_tlak_perm,M2_area, M2_typ, M2_tlak_perm, M3_area, M3_typ, C1_tlak, C2_tlak);
[ref, toky] =  simulator(x0(1),x0(2),x0(3),x0(4),x0(5),x0(6),x0(7),x0(8),x0(9),x0(10));
disp(ref);

str = sprintf('Power = %3.2f, Area = %3.2f, Recovery = %4.2f, Purity = %4.2f',ref(1), ref(2), -ref(3),-ref(4));

hold on
plot([0,120],[100,100],'k-'); plot([60,60],[50,100],'k-');
plot([20,80],[120,120],'k-'); plot([80,80],[120,100],'k-');
plot([20,20],[120,50],'k-') ; plot([20,120],[50,50],'k-');
plot([10,10],[100,100],'k>','markersize',12,'linewidth',2);
plot([30,30],[100,100],'k>','markersize',12,'linewidth',2);
plot([50,50],[100,100],'k>','markersize',12,'linewidth',2);
plot([70,70],[100,100],'k>','markersize',12,'linewidth',2);
plot([90,90],[100,100],'k>','markersize',12,'linewidth',2);
plot([120,120],[100,100],'k>','markersize',12,'linewidth',2);
plot([50,50],[120,120],'k>','markersize',12,'linewidth',2);
plot([80,80],[50,50],'k>','markersize',12,'linewidth',2);
plot([120,120],[50,50],'k>','markersize',12,'linewidth',2);
plot([40,40],[50,50],'k<','markersize',12,'linewidth',2);
plot([20,20],[60,60],'k^','markersize',12,'linewidth',2);
plot([20,20],[85,85],'k^','markersize',12,'linewidth',2);
plot([20,20],[110,110],'k^','markersize',12,'linewidth',2);
plot([60,60],[85,85],'kv','markersize',12,'linewidth',2);
plot([60,60],[60,60],'kv','markersize',12,'linewidth',2);
plot([80,80],[110,110],'kv','markersize',12,'linewidth',2);

mrkrsz = 30;

plot(0,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(0,100,'Com','HorizontalAlignment', 'center');
plot(20,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(20,100,'S1','HorizontalAlignment', 'center');
plot(40,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(40,100,'C1','HorizontalAlignment', 'center');
plot(60,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(60,100,'M1','HorizontalAlignment', 'center');
plot(80,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(80,100,'M3','HorizontalAlignment', 'center');
plot(100,100,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(100,100,'C3','HorizontalAlignment', 'center');
plot(20,75,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(20,75,'C5','HorizontalAlignment', 'center');
plot(60,75,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(60,75,'C2','HorizontalAlignment', 'center');
plot(60,50,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(60,50,'M2','HorizontalAlignment', 'center');
plot(100,50,'ks','MarkerSize',mrkrsz,'MarkerFaceColor','w'); text(100,50,'C4','HorizontalAlignment', 'center');

text(10,115,['p=',num2str(toky.Comb_S_p)],'HorizontalAlignment', 'center')
text(10,110,['CO2=',num2str(toky.Comb_S_CO2)],'HorizontalAlignment', 'center')
text(10,105,['N2=',num2str(toky.Comb_S_N2)],'HorizontalAlignment', 'center')
text(30,115,['p=',num2str(toky.S_C1_p)],'HorizontalAlignment', 'center')
text(30,110,['CO2=',num2str(toky.S_C1_CO2)],'HorizontalAlignment', 'center')
text(30,105,['N2=',num2str(toky.S_C1_N2)],'HorizontalAlignment', 'center')
text(50,115,['p=',num2str(toky.C1_M1_p)],'HorizontalAlignment', 'center')
text(50,110,['CO2=',num2str(toky.C1_M1_CO2)],'HorizontalAlignment', 'center')
text(50,105,['N2=',num2str(toky.C1_M1_N2)],'HorizontalAlignment', 'center')
text(70,115,['p=',num2str(toky.M1_M3_p)],'HorizontalAlignment', 'center')
text(70,110,['CO2=',num2str(toky.M1_M3_CO2)],'HorizontalAlignment', 'center')
text(70,105,['N2=',num2str(toky.M1_M3_N2)],'HorizontalAlignment', 'center')
text(90,115,['p=',num2str(toky.M3_C3_p)],'HorizontalAlignment', 'center')
text(90,110,['CO2=',num2str(toky.M3_C3_CO2)],'HorizontalAlignment', 'center')
text(90,105,['N2=',num2str(toky.M3_C3_N2)],'HorizontalAlignment', 'center')
text(110,115,['p=',num2str(toky.C3_OUT1_p)],'HorizontalAlignment', 'center')
text(110,110,['CO2=',num2str(toky.C3_OUT1_CO2)],'HorizontalAlignment', 'center')
text(110,105,['N2=',num2str(toky.C3_OUT1_N2)],'HorizontalAlignment', 'center')
text(50,135,['p=',num2str(toky.M3_S1_p)],'HorizontalAlignment', 'center')
text(50,130,['CO2=',num2str(toky.M3_S1_CO2)],'HorizontalAlignment', 'center')
text(50,125,['N2=',num2str(toky.M3_S1_N2)],'HorizontalAlignment', 'center')
text(25,95,['p=',num2str(toky.C5_S1_p)],'HorizontalAlignment', 'center')
text(25,90,['CO2=',num2str(toky.C5_S1_CO2)],'HorizontalAlignment', 'center')
text(25,85,['N2=',num2str(toky.C5_S1_N2)],'HorizontalAlignment', 'center')
text(40,65,['p=',num2str(toky.M2_C5_p)],'HorizontalAlignment', 'center')
text(40,60,['CO2=',num2str(toky.M2_C5_CO2)],'HorizontalAlignment', 'center')
text(40,55,['N2=',num2str(toky.M2_C5_N2)],'HorizontalAlignment', 'center')
text(80,65,['p=',num2str(toky.M2_C4_p)],'HorizontalAlignment', 'center')
text(80,60,['CO2=',num2str(toky.M2_C4_CO2)],'HorizontalAlignment', 'center')
text(80,55,['N2=',num2str(toky.M2_C4_N2)],'HorizontalAlignment', 'center')
text(65,95,['p=',num2str(toky.M1_C2_p)],'HorizontalAlignment', 'center')
text(65,90,['CO2=',num2str(toky.M1_C2_CO2)],'HorizontalAlignment', 'center')
text(65,85,['N2=',num2str(toky.M1_C2_N2)],'HorizontalAlignment', 'center')
text(65,70,['p=',num2str(toky.C2_M2_p)],'HorizontalAlignment', 'center')
text(65,65,['CO2=',num2str(toky.C2_M2_CO2)],'HorizontalAlignment', 'center')
text(65,60,['N2=',num2str(toky.C2_M2_N2)],'HorizontalAlignment', 'center')
text(110,65,['p=',num2str(toky.C4_OUT2_p)],'HorizontalAlignment', 'center')
text(110,60,['CO2=',num2str(toky.C4_OUT2_CO2)],'HorizontalAlignment', 'center')
text(110,55,['N2=',num2str(toky.C4_OUT2_N2)],'HorizontalAlignment', 'center')

axis([0,120,50,140]); title(str);
%axis equal
axis off