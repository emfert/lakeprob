% % This reproduces the lake problem model in Lempert and Collins (2007)

% function [Sopt, deltaLopt, L0opt] = Lempert_LakeProbOpt()
%     [Sopt, deltaLopt, L0opt] = [1,2,3];
% end

% function V = LCValFcn(S, deltaL, L0)
clear
S = 2.8;
deltaL = .27;
L0 = .31;


Pcrit = [.2 .7];
B = .2;              % decay rate of P concentration (B in lempert paper)
b = .1;                 % natural baseline loading
r = .25;                % P recycling parameter
dlta = 1/1.03;          % discount factor
alphaa = 1;             % relative marginal utility of loadings
omga = .04;             % st dev of stochastic shock
bbeta = 10;             % eutrophic cost
phi = 10;               % emissions reduction cost constant
gmma = .05;             % stdev of observed Xcrit
lmbda = .1;             % distance from xcrit req for learning
q = 2;                  % noise exponent

N = 2000;             % no. samples total, for initial data collection

NPt = 20;%41;               % no. grid points for Pt (concentration)
Npii = 20;%41;              % no. grid points for pii (probabilities)
Nlt = 20;%161;              % no. grid points for P loadings

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
lt = linspace(0,.8,Nlt);
T = 1000;                 % time span

Xcrit = [.3 .4 .5 .6 .7 .8 .9];
PXcrit = [0 0 0 .05 .25 .45 .25];

X0 = b/(1-B);
Xcrit_est0 = .787;
nu0 = .2;

% initialize loadings
L = zeros(N,length(Xcrit),T);
L(:,:,1) = L0;

% initialize estimates of Xcrit
Xcrit_est = zeros(N,length(Xcrit),T);
Xcrit_est(:,:,1) = Xcrit_est0;

% initialize concentrations
Xt = zeros(N,length(Xcrit),T);
Xt(:,:,1) = X0;

% initialize array to store simulated utility streams
U = zeros(N,length(Xcrit),T);
for i = 1:length(Xcrit)
    U(:,i,1) = alphaa*L0 - bbeta*(X0>Xcrit(i));
end

% do Monte Carlo sim

for i = 1:N % for each model
    for j = 1:length(Xcrit) % under each model
        nu_old = nu0;
        for k = 2:T % for the whole timespan
            
            bt = randn*omga + b;
            Xt(i,j,k) = B*Xt(i,j,k-1) + bt + L(i,j,k-1) + r*(Xt(i,j,k-1)>=Xcrit(j));
            
            gmmat = gmma*(Xt(i,j,k)>=Xcrit(j)) + gmma*exp(((Xcrit(j) - Xt(i,j,k))/lmbda)^2)*(Xt(i,j,k)<Xcrit(j));
            nu = nu_old*gmmat/(nu_old+gmmat);
            Zt = randn*gmma+Xcrit(j);
            Xcrit_est(i,j,k) = Xcrit_est(i,j,k-1) + (nu/(nu+gmmat))*(Zt - Xcrit_est(i,j,k-1));
            
            Ltarg = ((1-B)*Xcrit_est(i,j,k) - b - S*omga)*(Xt(i,j,k)<Xcrit(j)) ...
                + (Xcrit_est(i,j,k) - B*Xt(i,j,k) - r - b - S*omga)*(Xt(i,j,k)>=Xcrit(j));
            L(i,j,k) = max(0,(min(L(i,j,k-1)+deltaL,Ltarg)));
            
            U(i,j,k) = alphaa*L(i,j,k) - bbeta*(Xt(i,j,k)>=Xcrit(j)) - phi*max(L(i,j,k-1) - L(i,j,k),0);
            
            nu_old = nu;
            
        end
    end
end
Vp = zeros(N,length(Xcrit));
for i = 1:length(Xcrit)
    Vdum = squeeze(U(:,i,:));
    Vp(:,i) = Vdum*dlta.^(0:T-1)';
end

Vp2 = mean(Vp,1);
V = Vp2*PXcrit';

meanL = squeeze(mean(L,1));
meanXcrit_est = squeeze(mean(Xcrit_est,1));
meanXt = squeeze(mean(Xt,1));

% end

%% test figures

figure
ind = 7;
plot((0:T-1)/1,meanL(ind,:))
ylim([0 1])
xlim([0 100])
hold on
plot((0:T-1)/1,meanXcrit_est(ind,:))
plot((0:T-1)/1,meanXt(ind,:))
grid on

            
            
            
            
            
            
            
            
            