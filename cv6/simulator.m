function [out,toky] = simulator(M1_area,M1_typ, M1_tlak_perm,M2_area, M2_typ, M2_tlak_perm, M3_area, M3_typ, C1_tlak, C2_tlak)

M1_area = M1_area*1000; M1_tlak_perm = M1_tlak_perm*1000; M2_area = M2_area*1000;
M2_tlak_perm = M2_tlak_perm*1000;  M3_area = M3_area*1000; C1_tlak = C1_tlak*1000; C2_tlak = C2_tlak*1000;

Comb_S_p = 1e5;
Comb_S_CO2 = 1.375;
Comb_S_N2 = 13.9;

C5_S1_p = 100000;
C5_S1_CO2 = 0;
C5_S1_N2 = 0;

M3_S1_p = 100000;
M3_S1_CO2 = 0;
M3_S1_N2 = 0;

M1_typ = round(M1_typ); M2_typ = round(M2_typ); M3_typ = round(M3_typ); 

for i=1:10

S_C1_p = 1e5;
S_C1_CO2 = Comb_S_CO2 + C5_S1_CO2 + M3_S1_CO2;
S_C1_N2 = Comb_S_N2 + C5_S1_N2 + M3_S1_N2;

C1_M1_p = C1_tlak;
C1_M1_CO2 = S_C1_CO2;
C1_M1_N2 = S_C1_N2;

switch M1_typ
    case 1
        M1_CO2 = 5.70047E-09;
        temp = M1_CO2/13;
    case 2
        M1_CO2 = 7.50062E-09;
        temp = M1_CO2/50;
    case 3
        M1_CO2 = 1.50012E-08;
        temp = M1_CO2/49;
    case 4
        M1_CO2 = 1.11009E-08;
        temp = M1_CO2/55;
    case 5
        M1_CO2 = 2.4377E-08;
        temp = M1_CO2/12;
end

[valCO2,valN2] = Membrana(C1_M1_CO2,C1_M1_N2,C1_tlak,M1_tlak_perm,M1_CO2,temp,M1_area,0);

M1_M3_p = C1_tlak;
M1_M3_CO2 = S_C1_CO2-valCO2;
M1_M3_N2 = C1_M1_N2-valN2;

M1_C2_p = M1_tlak_perm;
M1_C2_CO2 = valCO2;
M1_C2_N2 = valN2;

%temp = (0.005667*(M3_CO2)^(-4.409))^(-1/3.409);

switch M3_typ
    case 1
        M3_CO2 = 5.70047E-09;
        temp = M3_CO2/13;
    case 2
        M3_CO2 = 7.50062E-09;
        temp = M3_CO2/50;
    case 3
        M3_CO2 = 1.50012E-08;
        temp = M3_CO2/49;
    case 4
        M3_CO2 = 1.11009E-08;
        temp = M3_CO2/55;
    case 5
        M3_CO2 = 2.4377E-08;
        temp = M3_CO2/12;
end

[valCO2,valN2] = Membrana(M1_M3_CO2,M1_M3_N2,M1_M3_p,100000,M3_CO2,temp,M3_area,Comb_S_CO2+Comb_S_N2);


M3_S1_p = 100000;
M3_S1_CO2 = valCO2;
M3_S1_N2 = valN2;

M3_C3_p = C1_tlak;
M3_C3_CO2 = M1_M3_CO2-valCO2;
M3_C3_N2 = M1_M3_N2-valN2;

C3_OUT1_p = 100000;
C3_OUT1_CO2 = M3_C3_CO2;
C3_OUT1_N2 = M3_C3_N2;

C2_M2_p = C2_tlak;
C2_M2_CO2 = M1_C2_CO2;
C2_M2_N2 = M1_C2_N2;

switch M2_typ
    case 1
        M2_CO2 = 5.70047E-09;
        temp = M2_CO2/13;
    case 2
        M2_CO2 = 7.50062E-09;
        temp = M2_CO2/50;
    case 3
        M2_CO2 = 1.50012E-08;
        temp = M2_CO2/49;
    case 4
        M2_CO2 = 1.11009E-08;
        temp = M2_CO2/55;
    case 5
        M2_CO2 = 2.4377E-08;
        temp = M2_CO2/12;
end

[valCO2,valN2] = Membrana(C2_M2_CO2,C2_M2_N2,C2_tlak,M2_tlak_perm,M2_CO2,temp,M2_area,0);

M2_C5_p = C2_tlak;
M2_C5_CO2 = C2_M2_CO2-valCO2;
M2_C5_N2 = C2_M2_N2-valN2;

C5_S1_p = 100000;
C5_S1_CO2 = M2_C5_CO2;
C5_S1_N2 = M2_C5_N2;

M2_C4_p = M2_tlak_perm;
M2_C4_CO2 = valCO2;
M2_C4_N2 = valN2;

C4_OUT2_p = 8000000;
C4_OUT2_CO2 = M2_C4_CO2;
C4_OUT2_N2 = M2_C4_N2;

x(i,1) = S_C1_N2;
x(i,2) = S_C1_CO2;

end

cost = 0; 
a1 = 1/(0.95*0.9); a2 =0.9/0.95;
cost = cost + a1*comp_compress(S_C1_CO2,S_C1_N2,S_C1_p,C1_M1_p);
cost = cost + a1*comp_compress(M1_C2_CO2,M1_C2_N2,M1_C2_p,C2_M2_p);
cost = cost + a1*comp_compress(M2_C4_CO2,M2_C4_N2,M2_C4_p,C4_OUT2_p);
cost = cost + a2*comp_expans(M3_C3_CO2,M3_C3_N2,M3_C3_p,C3_OUT1_p);
cost = cost + a2*comp_expans(M2_C5_CO2,M2_C5_N2,M2_C5_p,C5_S1_p);

cost = cost/1e6;

CO2_rec = C4_OUT2_CO2/Comb_S_CO2; 
CO2_rec = CO2_rec*100;

CO2_pur = C4_OUT2_CO2/(C4_OUT2_CO2+C4_OUT2_N2);
CO2_pur = CO2_pur*100;

total_area = M1_area+M2_area+M3_area; 
total_area = total_area/1000;

if M3_S1_CO2 < 0 || C4_OUT2_CO2 < 0 || C4_OUT2_N2 < 0 || CO2_rec > 100 || isnan(CO2_rec) || isnan(CO2_pur) || CO2_pur < 50
    out = [inf,inf,inf,inf];
else
    out = [cost,total_area,-CO2_rec,-CO2_pur];
end

toky.C1_M1_CO2 = C1_M1_CO2; toky.C1_M1_N2 = C1_M1_N2; toky.C1_M1_p = C1_M1_p; toky.C1_tlak = C1_tlak; 
toky.C2_M2_CO2 = C2_M2_CO2; toky.C2_M2_N2 = C2_M2_N2; toky.C2_M2_p = C2_M2_p; toky.C2_tlak = C2_tlak; 
toky.C3_OUT1_CO2 = C3_OUT1_CO2; toky.C3_OUT1_N2 = C3_OUT1_N2; toky.C3_OUT1_p = C3_OUT1_p; 
toky.C4_OUT2_CO2 = C4_OUT2_CO2; toky.C4_OUT2_N2 = C4_OUT2_N2; toky.C4_OUT2_p = C4_OUT2_p; 
toky.C5_S1_CO2 = C5_S1_CO2; toky.C5_S1_N2 = C5_S1_N2; toky.C5_S1_p = C5_S1_p; 
toky.CO2_pur = CO2_pur; toky.CO2_rec = CO2_rec;
toky.Comb_S_CO2 = Comb_S_CO2; toky.Comb_S_N2 = Comb_S_N2; toky.Comb_S_p = Comb_S_p; 
toky.M1_area = M1_area; toky.M1_C2_CO2 = M1_C2_CO2; toky.M1_C2_N2 = M1_C2_N2; toky.M1_C2_p = M1_C2_p; 
toky.M1_CO2 = M1_CO2; toky.M1_M3_CO2 = M1_M3_CO2; toky.M1_M3_N2 = M1_M3_N2; toky.M1_M3_p = M1_M3_p; toky.M1_tlak_perm = M1_tlak_perm; 
toky.M2_area = M2_area; toky.M2_C4_CO2 = M2_C4_CO2; toky.M2_C4_N2 = M2_C4_N2; toky.M2_C4_p = M2_C4_p; 
toky.M2_C5_CO2 = M2_C5_CO2; toky.M2_C5_N2 = M2_C5_N2; toky.M2_C5_p = M2_C5_p; toky.M2_CO2 = M2_CO2; toky.M2_tlak_perm = M2_tlak_perm; 
toky.M3_area = M3_area; toky.M3_C3_CO2 = M3_C3_CO2; toky.M3_C3_N2 = M3_C3_N2; toky.M3_C3_p = M3_C3_p;
toky.M3_CO2 = M3_CO2; toky.M3_S1_CO2 = M3_S1_CO2; toky.M3_S1_N2 = M3_S1_N2; toky.M3_S1_p = M3_S1_p; 
toky.S_C1_CO2 = S_C1_CO2; toky.S_C1_N2 = S_C1_N2; toky.S_C1_p = S_C1_p; 
end


function [V_CO2,V_N2] = Membrana(Vf_CO2,Vf_N2,pr,pp,P_CO2,P_N2,S,Vs_N2)

Vp_CO2_prev=0;Vp_N2_prev=0;
Vr_CO2=Vf_CO2;Vr_N2=Vf_N2;

x=20;
y=1000;

for i=1:y
    if Vs_N2>0
        pp_CO2=0;pp_N2=pp;
    else
        pp_CO2=0;pp_N2=0;
    end
    for j = 1:x
        deltap_CO2=pr*(Vr_CO2/(Vr_CO2+Vr_N2+1E-9))-pp_CO2;
        deltap_N2=pr*(Vr_N2/(Vr_CO2+Vr_N2+1E-9))-pp_N2;
        Vp_CO2 = P_CO2*S/y*deltap_CO2+Vp_CO2_prev;
        Vp_N2 = P_N2*S/y*deltap_N2+Vp_N2_prev;
        pp_CO2=pp*(Vp_CO2/(Vp_CO2+Vp_N2+Vs_N2+1E-9));
        pp_N2=pp*((Vp_N2+Vs_N2)/(Vp_CO2+Vp_N2+Vs_N2+1E-9));
    end
    if (P_CO2*S/y*deltap_CO2) >= (0.5*Vr_CO2)
        Vp_CO2=0.5*Vr_CO2+Vp_CO2_prev;
    end
    if (P_CO2*S/y*deltap_CO2) < 0
        Vp_CO2=Vp_CO2_prev;
    end
    if (P_N2*S/y*deltap_N2) >= (0.5*Vr_N2)
        Vp_N2=0.5*Vr_N2+Vp_N2_prev;
    end
    if ((P_N2*S/y*deltap_N2) < 0) && (Vs_N2==0)
        Vp_N2=Vp_N2_prev;
    end
    Vp_CO2_prev=Vp_CO2;Vp_N2_prev=Vp_N2;
    Vr_CO2=Vf_CO2-Vp_CO2;Vr_N2=Vf_N2-Vp_N2;
end
V_CO2 = Vp_CO2;V_N2 = Vp_N2;
end



function [W] = comp_compress(V_CO2,V_N2,p1,p2)
T = 323.15; R=8.314462; V_M=0.022414; x=1.3;

n   = (V_CO2 + V_N2)/V_M;
V1  = T*R*n/p1;
V15 = V1*((p1/p2)^(1/x));
W1  = (p2*V15 - p1*V1)/(x-1);
T15 = T*p2*V15/p1/V1;
W2  = n*R*(T15-T);

W = W1+W2;

end


function [W] = comp_expans(V_CO2,V_N2,p1,p2)
T = 323.15; R=8.314462; V_M=0.022414; x=1.3;

n   = (V_CO2 + V_N2)/V_M;
V1  = T*R*n/p1;
V15 = V1*((p1/p2)^(1/x));
W1  = (p2*V15 - p1*V1)/(x-1);
T15 = T*p2*V15/p1/V1;
W2  = n*R*(T15-T);

W = W1+W2;
end


