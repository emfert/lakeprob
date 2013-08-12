% % This reproduces the lake problem model in Lempert and Collins (2007)

function [x,fval,exitflag] = Lempert_LakeProbOpt()
    x0 = [2.5, .3, .3];
    options = optimset('Display','iter','TolFun',1e-3,'TolX',1e-3,'MaxIter',100);
    [x,fval,exitflag] = fminsearch(@LCValFcn,x0,options);
end

function V = LCValFcn(x)
% clear
% S = 2.8;
% deltaL = .27;
% L0 = .31;
S = x(1);
deltaL = x(2);
L0 = x(3);

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
T = 100;                 % time span

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

rng(2) % seed RNG
% do Monte Carlo sim

for i = 1:N % for each model
    for j = 1:length(Xcrit) % under each model
        nu_old = nu0;
        for k = 2:T % for the whole timespan
            
            bt = randn*omga + b;
            Xt(i,j,k) = B*Xt(i,j,k-1) + bt + L(i,j,k-1) + r*(Xt(i,j,k-1)>=Xcrit(j));
            
            gmmat = gmma*(Xt(i,j,k)>=Xcrit(j)) + gmma*exp(((Xcrit(j) - Xt(i,j,k))/lmbda)^q)*(Xt(i,j,k)<Xcrit(j));
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
V = -Vp2*PXcrit'; % made this negative because we're maximizing

%V.meanL = squeeze(mean(L,1));
%V.meanXcrit_est = squeeze(mean(Xcrit_est,1));
%V.meanXt = squeeze(mean(Xt,1));

end

% %% test figures
% 
% figure
% ind = 7;
% plot((0:T-1)/1,meanL(ind,:))
% ylim([0 1])
% xlim([0 100])
% hold on
% plot((0:T-1)/1,meanXcrit_est(ind,:))
% plot((0:T-1)/1,meanXt(ind,:))
% grid on

            
            
            
            
            
            
            
            
            