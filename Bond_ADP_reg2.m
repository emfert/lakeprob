% This implements the lake problem as done in the Lempert paper, using a
% linear model. Unlike previous versions, it keeps track of the previous
% loading rate as a state.

% This newer version has more flexibility in basis functions and uses
% Gauss-Hermite quadrature to approximate the normal PDF rather than the
% three-point method

function results = Bond_ADP_reg2()
% set up initial parameters
clear
Pcrit = [.3 .6];     % candidate eutrophication thresholds
B = .2;                 % decay rate of P concentration (B in lempert paper)
b = .1;                 % natural baseline loading
r = .25;                % P recycling parameter
dlta = 1/1.03;          % discount factor (DR = ((1/dlta)-1) * 100%)
alphaa = 1;             % relative marginal utility of loadings
sgma = .04;             % st dev of stochastic shock
bbeta = 10;             % eutrophic cost
phi = 10;               % emissions reduction cost constant

N = 30;                % no. samples total, for initial data collection
p = .05;                  % probability it makes a random decision instead of optimal

% for Gauss-Hermite quadrature
xk_h = [-2.65196 -1.67355 -.81629 0 .81629 1.67355 2.65196];
xk = sqrt(2)*sgma*xk_h;
wxk = [.0009718 .0545 .4256 .8103 .4256 .0545 .0009718];
g = length(xk_h);                  % no. points for Gauss-Hermite quadrature

%pct5 = norminv(.05,0,sgma);
%pct95 = norminv(.95,0,sgma);

NPt = 41;               % no. grid points for Pt (concentration)
Npii = 41;              % no. grid points for pii (probabilities)
Nlt = 81;               % no. grid points for P loadings

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
lt = linspace(0,.8,Nlt);
T = 20;                 % time span

%% sample points to fit initial regression

% generate value function approximation coefficients
int = 5;    % intercept
ltc = -2;   % previous time step's loading rate
ptc = -1;   % concentration
piic = -3;  % threshold exceedance

V = @(pt) [int ltc ptc piic*ones(1,length(Pcrit)-1)]*pt';

% Sobol sample for N initial concentrations and prob dists
pp = sobolset(2+length(Pcrit));
testpts = net(pp,N);
testpts(:,1) = testpts(:,1)*lt(end);
testpts(:,2) = testpts(:,2)*Pt(end);
for i = 4:length(Pcrit)+1
    testpts(:,i) = testpts(:,i).*(1-sum(testpts(:,3:i-1),2));
end
testpts(:,end) = 1-sum(testpts(:,3:end-1),2);

% initialize results array
res = zeros(length(Pcrit)+4,T,N);

for n = 1:N
    n
    S = testpts(n,2);                   % initial concentration
    
    % shuffle prob vectors so that distributions are about the same
    % (without this, the first sample is uniform and the others have too
    % much mass at low values
    testpts(n,3:end) = testpts(n,2+randperm(length(Pcrit)));
    P = testpts(n,3:end);               % initial prob dist
    
    lt_prev = testpts(n,1);             % initial previous loading rate
    
    for t = 1:T-1
        Vdum = zeros(1,Nlt);
        for k = 1:Nlt               % iterate through control space                        
            % calculate current utility
            U = alphaa*lt(k) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(k))*(lt_prev>lt(k));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each candidate
            % threshold
            
            %pts = zeros(g,length(Pcrit));
            pts_mid = B*S + b + lt(k) + r*(S>Pcrit);
            pts_help = kron(ones(g,1),pts_mid);
            xk_help = kron(ones(1,length(Pcrit)),xk');
            pts = pts_help+xk_help;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(g,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                Lt(:,:,i) = exp(-(pts - pts_mid(i)).^2)/(2*sgma^2);
            end
            
            % generate array that's no. test points for calculating EV for
            % each model X number of models to test X number of predictors
            piplus = zeros(g,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,i,:))*P';
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,i,:) = (squeeze(Lt(:,i,:)).*kron(ones(g,1),P))./pi_help2;
            end
            
            % approximate V(t+1) for each point necessary for EV calculation
            Vpts = zeros(g,length(Pcrit));
            for i = 1:g
                for j = 1:length(Pcrit)
                    Vpts(i,j) = V([1 lt(k) pts(i,j) squeeze(piplus(i,j,1:end-1))']);
                end
            end

            % EV of V(t+1)
            E = (1/sqrt(pi))*wxk*Vpts;            
            Vdum(k) = U + dlta*(E*P');
        end
        
        % find optimal (or other) loading rate
        if rand <= p            % make a random decision to explore space
            Vnthelp = Vdum(~isnan(Vdum));
            Vnthelp2 = randperm(length(Vnthelp));
            Vnt = Vnthelp(Vnthelp2(1));
        else
            Vnt = max(Vdum);
        end
        
        % record decisions and function values
        ltdum = lt(Vnt==Vdum);
        U = alphaa*ltdum - bbeta*(S>Pcrit)*P' - phi*(lt_prev-ltdum)*(lt_prev>ltdum);
        
        res(1,t,n) = lt_prev;
        res(2,t,n) = S;
        res(3:2+length(Pcrit),t,n) = P;
        res(3+length(Pcrit),t,n) = U;
        res(4+length(Pcrit),t,n) = ltdum;
        res(5+length(Pcrit),t,n) = Vnt;
        
        % update lt_prev, concentration S, and probdist P
        lt_prev = ltdum;
        
        %p_log = zeros(length(Pcrit),1);
        %for i = 1:length(Pcrit)
        %    p_log(i) = S>Pcrit(i);
        %end        
        Sdum = B*S + b + ltdum + r*P*(S>Pcrit)' + randn*sgma;
        
        Ltb = exp(-(Sdum - (B*S + b + ltdum + r*(S>Pcrit)')).^2)/(2*sgma^2);        
        P = P.*Ltb'/(P*Ltb);
        
        % no nonphysical concentrations
        if Sdum < 0
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = Sdum;
        end
    end
end
%% or just load points from a workspace, if not sampling - run top cell, then...
% load BondADP10k
% V = results.V;

% initialize matrix to store regression coefficients
% intguess = [0 0 0 0 0];           % for testing interactions, guess for interaction parameters
intguess = [];

num_int = length(intguess);
coefmat = zeros(T,length(Pcrit)+1,length(Pcrit)+2+num_int); % time, plane, param
coefmat(end,:,:) = kron(ones(length(Pcrit)+1,1),[5 -1 -2 -3*ones(1,length(Pcrit)-1) intguess]);

planetest = [0 Pcrit 1];
for t = 1:T-1
    % do interactions - this is previous loading rate with both
    % concentration and probability of the lowest threshold
    inthelp = squeeze(res(:,t,:))';
    int = [inthelp(:,1).*inthelp(:,2) inthelp(:,1).*inthelp(:,3) ...
        inthelp(:,1).^2 inthelp(:,2).^2 inthelp(:,3).^2];
    
    for i = 1:length(Pcrit)+1
        regvecx = squeeze(res(1:length(Pcrit)+1,t,(res(2,t,:)>=planetest(i))&(res(2,t,:)<planetest(i+1))))';
        
        % for any interactions:
        % intx = int((res(2,t,:)>=planetest(i))&(res(2,t,:)<=planetest(i+1)),:);
        
        % for no interactions:
        intx = [];
        
        regvecx = [ones(size(regvecx,1),1) regvecx intx];
        regvecy = squeeze(res(end,t,(res(2,t,:)>=planetest(i))&(res(2,t,:)<planetest(i+1))));
        coefmat(t,i,:) = regress(regvecy,regvecx);
    end
end
coefmatold = coefmat;

% plot function values from lookup table
% figure
% surf(Pt,pii,squeeze(Vhelp(:,:,1))')
% view(142.5, 30)
% xlabel('Concentration')
% ylabel('Probability')
% zlabel('Ob fun value')
% title('Lookup table values')
%saveas(gcf,'../../../Desktop/lakeproblem/lookup0','epsc')

%% ADP stage

M = 10;                % number of ADP iterations
results = [];

%intA = '[lt(i)*pts(j,k) lt(i)*piplus(j,k,1) lt(i)^2 pts(j,k)^2 piplus(j,k,1)^2]';
%intB = '[lt_prev*S; lt_prev*P(1); lt_prev^2; S^2; P(1)^2]';

intA = '[]';
intB = '[]';

for m = 1:M
    m
    % sample random concentration, probability distribution
    S = rand*Pt(end);  
    
    dif = 0;            
    P = zeros(1,length(Pcrit));
    for i = 1:length(Pcrit)-1
        P(i) = rand*(1-dif);
        dif = dif+P(i);
    end
    P(end) = 1-dif;
    P = P(randperm(length(P)));
    
    % initial prior loading rate
    lt_prev = rand*lt(end);      
    
    for t = 1:T-1
        
        % evaluate functions for each loading rate
        Vdum = zeros(1,Nlt);
        for i = 1:Nlt
            U = alphaa*lt(i) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(i))*(lt_prev>lt(i));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each threshold
            
            pts_mid = B*S + b + lt(k) + r*(S>Pcrit);
            pts_help = kron(ones(g,1),pts_mid);
            xk_help = kron(ones(1,length(Pcrit)),xk');
            pts = pts_help+xk_help;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(g,length(Pcrit),length(Pcrit));
            for j = 1:length(Pcrit)
                Lt(:,:,j) = exp(-(pts - pts_mid(j)).^2)/(2*sgma^2);
            end
            
            % make array that's no. test points for calculating EV for
            % each model X number of models to test X number of predictors
            piplus = zeros(g,length(Pcrit),length(Pcrit));
            for j = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,j,:))*P';
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,j,:) = (squeeze(Lt(:,j,:)).*kron(ones(g,1),P))./pi_help2;
            end   
            
            % make a matrix with appropriate regression coefficients for
            % t+1 concentrations
            coefmat2 = zeros(g,length(Pcrit),length(Pcrit)+2+num_int);
            for j = 1:g
                for k = 1:length(Pcrit)
                    for jj = 1:length(planetest)-1
                        if (pts(j,k)>=planetest(jj))&&(pts(j,k)<=planetest(jj+1))
                            coefmat2(j,k,:) = squeeze(coefmat(t+1,jj,:));
                        end
                    end
                end
            end
                        
            % calculate value function for t+1
            Vtp1 = zeros(size(pts));
            for j = 1:g
                for k = 1:length(Pcrit)
                    Vtp1(j,k) = squeeze(coefmat2(j,k,:))'*[1 lt(i) pts(j,k) squeeze(piplus(j,k,1:end-1))' ... 
                        eval(intA)]';
                end
            end
                        
            % calculate EV for t+1
            Vdum(i) = U + dlta*(1/sqrt(pi))*wxk*Vtp1*P';
        end
        
        % sometimes use a random loading rate instead of the optimal one
        if rand <= p
            randdum2 = randperm(Nlt);
            ltdum = lt(randdum2(1));
            Vnew = Vdum(randdum2(1));
        else % just take optimal value
            ltdum = lt(Vdum==max(Vdum));
            Vnew = max(Vdum);
        end
        
        % figure out which regression plane you're on and calculate error
        for j = 1:length(planetest)-1
            if (S>=planetest(j))&&(S<=planetest(j+1))
                whichplane = j;
            end
        end
        
        error = Vnew - squeeze(coefmat(t,whichplane,:))'*[1; lt_prev; S; P(1:end-1)'; eval(intB)];
        
        % calculate gradient
        grad = zeros(1,1,2+length(Pcrit)+num_int);
        grad(:,:,:) = [-1; -lt_prev; -S; -P(1:end-1)'; -eval(intB)];
        
        % choose step size
        alfa = 1/(m+N);
        alfamult = 10;  %experiment with changing its size
        alfa = alfa*alfamult;
        
        % update regression coefficient
        coefmat(t,whichplane,:) = coefmat(t,whichplane,:) - alfa*error*grad;
        
        % store initial state, probability, loading rate, function value,
        % and coefficient estimates for chosen time (just for
        % debugging/tuning)
        if t==5
            results.new(m,1) = Vnew;
            results.new(m,2) = error;
            results.new(m,3) = lt_prev;
            results.new(m,4) = ltdum;
            results.new(m,5) = S;
            results.new(m,6:6+length(Pcrit)-1) = P';            
            results.coefupd(m,:,:) = coefmat(t,:,:);
        end
        
        % store this period's loading rate for next time step
        lt_prev = ltdum;
        
        % simulate concentration in next period      
        Sdum = B*S + b + ltdum + r*P*(S>Pcrit)' + randn*sgma;
        
        % update probability distribution given simulated concentration
        Ltb = exp(-(Sdum - (B*S + b + ltdum + r*(S>Pcrit)')).^2)/(2*sgma^2);        
        P = P.*Ltb'/(P*Ltb);
        
        % no nonphysical concentrations
        if Sdum < 0     % update concentration for next timestep
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = Sdum;
        end
        
    end
end

results.V = V;
results.Pt = Pt;
results.lt = lt;
results.pii = pii;
results.coefmat = coefmat;
results.coefmatold = coefmatold;

% %% diagnostic plots
% 
% %coefmat = results.coefmat;
% 
% % final regression planes
% f1 = @(x,y) squeeze(coefmat(1,1,:))'*[1; x; y];
% f2 = @(x,y) squeeze(coefmat(1,2,:))'*[1; x; y];
% f3 = @(x,y) squeeze(coefmat(1,3,:))'*[1; x; y];
% figure
% hold on
% ezmesh(f1,[0 Pt(b1) 0 1])
% ezmesh(f2,[Pt(b1) Pt(b2) 0 1])
% ezmesh(f3,[Pt(b2) 1 0 1])
% xlim([0 1])
% zlim([0 5])
% xlabel('Concentration')
% ylabel('Probability')
% zlabel('Ob fun value')
% title('Final value function')
% view(142.5, 30)
% %saveas(gcf,'../../../Desktop/lakeproblem/finalval0','epsc')
% 
% % initial regression planes (regressed from lookup table)
% f1 = @(x,y) squeeze(coefmatold(1,1,:))'*[1; x; y];
% f2 = @(x,y) squeeze(coefmatold(1,2,:))'*[1; x; y];
% f3 = @(x,y) squeeze(coefmatold(1,3,:))'*[1; x; y];
% figure
% hold on
% ezmesh(f1,[0 Pt(b1) 0 1])
% ezmesh(f2,[Pt(b1) Pt(b2) 0 1])
% ezmesh(f3,[Pt(b2) 1 0 1])
% xlim([0 1])
% zlim([0 5])
% xlabel('Concentration')
% ylabel('Probability')
% zlabel('Ob fun value')
% title('Initial value function')
% view(142.5, 30)
% %saveas(gcf,'../../../Desktop/lakeproblem/initval0','epsc')
% 
% % parameter estimates as a function of iteration
% figure
% for i = 1:3
%     subplot(1,3,i)
%     hold on
%     plot(squeeze(results.coefupd(:,i,1)))
%     plot(squeeze(results.coefupd(:,i,2)),'r')
%     plot(squeeze(results.coefupd(:,i,3)),'g')
%     legend({'\beta_0' '\beta_1' '\beta_2'},'Location','East')
%     title(['Plane' num2str(i) 'parameters'])
% end
% %saveas(gcf,'../../../Desktop/lakeproblem/params0','epsc')
% 
% % optimal loading rate
% figure
% scatter3(results.new(:,1), results.new(:,2), results.new(:,3))
% xlabel('Concentration')
% ylabel('Probability')
% zlabel('Optimal loading rate')
% view(142.5,30)

%% do comparison with Lempert and Collins model

% seed rng

P0 = [.4 .6];        % initial guess for prob dist
%X0 = .5;%.787;              % initial state
X0 = b/(1-B);
Xcrit = Pcrit(2);             % actual Xcrit value
NN = 300;              % number of sample paths to simulate
T2 = 30;               % how far out in time to simulate them
L0 = 0;                 % initial loading rate

% get coefficient matrix for forward simulations, usu. t=1
coefmatsim = squeeze(coefmat(1,:,:));
%intC = '[L(i,t)*pts(jj,k) L(i,t)*piplus(jj,k,1) L(i,t)^2 pts(jj,k)^2 piplus(jj,k,1)^2]';
intC = '[]';

%ints = '[L0*X0 L0*P0(1) L0^2 X0^2 P0(1)^2]';
ints = '[]';

for jj = 1:length(planetest)-1
    if (X0>=planetest(jj))&&(X0<=planetest(jj+1))
        V_est = coefmatsim(jj,:)*[1 L0 X0 P0(1:end-1) eval(ints)]';
    end
end

% initialize concentration and loading storage matrices
X = zeros(NN,T2);
L = zeros(NN,T2);
P = zeros(NN,T2,length(Pcrit));
V = zeros(NN,T2);
U = zeros(NN,T2);

X(:,2) = X0;
P(:,2,:) = kron(ones(NN,1),P0);
L(:,1) = L0;

for i = 1:NN
    i
    for t = 2:T2
        % calculate optimal loading rate
        Vdum = zeros(Nlt,1);
        Udum = zeros(Nlt,1);
        for j = 1:Nlt
            Udum(j) = alphaa*lt(j) - bbeta*(X(i,t)>Xcrit) - phi*(L(i,t-1)-lt(j))*(L(i,t-1)>lt(j));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each threshold
            
            pts_mid = B*S + b + lt(k) + r*(S>Pcrit);
            pts_help = kron(ones(g,1),pts_mid);
            xk_help = kron(ones(1,length(Pcrit)),xk');
            pts = pts_help+xk_help;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(g,length(Pcrit),length(Pcrit));
            for k = 1:length(Pcrit)
                Lt(:,:,k) = exp(-(pts - pts_mid(k)).^2)/(2*sgma^2);
            end
            
            % make array that's no. test points for calculating EV for
            % each model X number of models to test X number of predictors
            piplus = zeros(g,length(Pcrit),length(Pcrit));
            for k = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,k,:))*squeeze(P(i,t,:));
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,k,:) = (squeeze(Lt(:,k,:)).*kron(ones(g,1),squeeze(P(i,t,:))'))./pi_help2;
            end   
            
            % make a matrix with appropriate regression coefficients for
            % t+1 concentrations
            for kk = 1:g
                for k = 1:length(Pcrit)
                    for jj = 1:length(planetest)-1
                        if (pts(kk,k)>=planetest(jj))&&(pts(kk,k)<=planetest(jj+1))
                            coefmat2 = coefmatsim(jj,:);
                        end
                    end
                end
            end
                        
            % calculate value function for t+1
            Vtp1 = zeros(size(pts));
            for jj = 1:g
                for k = 1:length(Pcrit)
                    Vtp1(jj,k) = coefmat2*[1 L(i,t) pts(jj,k) squeeze(piplus(jj,k,1:end-1))' eval(intC)]';
                end
            end
                        
            % calculate EV for t+1
            Vdum(j) = Udum(j) + dlta*(1/sqrt(pi))*wxk*Vtp1*squeeze(P(i,t,:));
        end
        
        L(i,t) = lt(Vdum==max(Vdum));
        V(i,t) = max(Vdum);
        U(i,t) = Udum(Vdum==max(Vdum));
        
        % simulate concentration in next period     
        X(i,t+1) = B*X(i,t) + b + L(i,t) + r*(X(i,t)>Xcrit) + randn*sgma;
        
        % update probability distribution given simulated concentration
        Ltb = exp(-(X(i,t+1) - (B*X(i,t) + b + L(i,t) + r*(X(i,t)>Pcrit))).^2)/(2*sgma^2);        
        P(i,t+1,:) = squeeze(P(i,t,:))'.*Ltb/(squeeze(P(i,t,:))'*Ltb');
        
        % no nonphysical concentrations
        if X(i,t+1) < 0
            X(i,t+1) = 0;
        elseif X(i,t+1) > 1
            X(i,t+1) = 1;
        else
        end                
    end
end

Vcalc = U*dlta.^(0:T2-1)';

%% diagnostic plots

indic = '[ltest*ss ltest*pp ltest^2 ss^2 pp^2]';

% value function
ltest = .05;
testfun1 = @(ss,pp) coefmatsim(1,:)*[1 ltest ss pp eval(indic)]';
testfun2 = @(ss,pp) coefmatsim(2,:)*[1 ltest ss pp eval(indic)]';
testfun3 = @(ss,pp) coefmatsim(3,:)*[1 ltest ss pp eval(indic)]';

figure
hold on
ezmesh(testfun1, [0, Pcrit(1), 0, 1])
ezmesh(testfun2, [Pcrit(1), Pcrit(2), 0, 1])
ezmesh(testfun3, [Pcrit(2), 1, 0, 1])
xlim([0 1])
zlim([-10 5])

%% plotting sample paths
figure
hold on
grid on
plot(X')

figure
hold on
grid on
plot(L')

figure
hold on
grid on
plot(squeeze(P(:,:,2))')




end
        
        
        
        
        
        
        
        
        
        