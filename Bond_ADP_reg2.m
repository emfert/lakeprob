% this is a crack at using ADP to implement the lake problem from the Bond
% paper
% unlike Bond_ADP, it uses a simulated annealing type thing as described in
% Powell to search for unexpected optima, it doesn't just go straight to
% teh best one
% now it adds the regression too

function results = Bond_ADP_reg2()
% set up initial parameters
clear
Pcrit1 = .2; % or .7    % critical threshold
Pcrit2 = .7;
gmma = .1;              % decay rate of P concentration
b = .02;                % natural baseline loading
r = .2;                 % P recycling parameter
dlta = .99;             % discount factor
bta = 1.5;              % relative marginal utility of loadings
sgma = .141421;         % st dev of stochastic shock
N = 5000; %NOT ENOUGH               % no. samples total, for initial data collection
p = 0;                % probabilit it jumps to a random decision

pct5 = norminv(.05,0,sgma);
pct95 = norminv(.95,0,sgma);

NPt = 41;               % no. grid points for Pt
Npii = 41;              % no. grid points for pii
Nlt = 161;              % no. grid points for P loadings
%Hn = 16;                % Hermite nodes and weights
%eps = .001;             % Value function error tolerance

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
lt = linspace(0,.8,Nlt);
T = 10;                 % time span

%% sample points to fit initial regression

V = zeros(NPt,Npii,T);
for i = 1:NPt
    V(i,:,end) = 3 - pii - Pt(i);      % find appropriate final condition
end
ltopt = zeros(NPt,Npii,T);
[X,Y] = meshgrid(pii,Pt');

% generate lookup table for first N samples
for n = 1:N
    n
    % initial state variables
    randdum = randperm(NPt);
    S = Pt(randdum(1));
    randdum2 = randperm(Npii);
    P = pii(randdum2(1));
    
    for t = 1:T-1
        Vdum = zeros(1,Nlt);
        for k = 1:Nlt            
            U = bta*lt(k) - S^2;
            
            % do EV calculation
            m1 = gmma*S + b + lt(k) + (S>Pcrit1)*r;
            p5_1 = m1+pct5;
            p95_1 = m1+pct95;
            m2 = gmma*S + b + lt(k) + (S>Pcrit2)*r;
            p5_2 = m2+pct5;
            p95_2 = m2+pct95;
            pts = [p5_1 m1 p95_1 p5_2 m2 p95_2];
            
            % likelihood functions and Bayesian updating
            Lt1 = exp(-(pts - pts(2)).^2/(2*sgma^2));
            Lt2 = exp(-(pts - pts(5)).^2/(2*sgma^2));
            piplus = P*Lt1./(P*Lt1 + (1-P)*Lt2);
            
            % do interpolation for Vtp1 (for lookup table)
            Vpts = interp2(X,Y,squeeze(V(:,:,t+1)),piplus,pts);
            
            E1 = .185*Vpts(1)+.63*Vpts(2)+.185*Vpts(3);
            E2 = .185*Vpts(4)+.63*Vpts(5)+.185*Vpts(6);
            
            Vdum(k) = U + dlta*(P*E1+(1-P)*E2);
        end
        if rand <= p            % make a random decision to explore space
            Vnthelp = Vdum(~isnan(Vdum));
            Vnthelp2 = randperm(length(Vnthelp));
            Vnt = Vnthelp(Vnthelp2(1));
        else
            Vnt = max(Vdum);
        end
        
        % put decisions and function values in lookup table
        ltdum = lt(Vnt==Vdum);
        V(S==Pt,P==pii,t) = Vnt;
        ltopt(S==Pt,P==pii,t) = ltdum;
        
        % calculate (random) state and probability estimate for next
        % timestep
        Sdum = gmma*S + b + ltdum + P*r*(S>Pcrit1) + (1-P)*r*(S>Pcrit2) + randn*sgma;       
        Lt1b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit1)*r))^2/(2*sgma^2));
        Lt2b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit2)*r))^2/(2*sgma^2));
        Pdum = P*Lt1b/(P*Lt1b + (1-P)*Lt2b);
        if Sdum < 0     % update concentration for next timestep
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = interp1(Pt,Pt,Sdum,'nearest');
        end
        P = interp1(pii,pii,Pdum,'nearest');    % update probability estimate
    end
end

%% or just load points from a workspace, if not sampling
load BondADP10k
V = results.V;
Pt = results.Pt;
pii = results.pii;

% find boundaries of planes
Vhelp = V;
meandV = zeros(NPt,1);
Vhelp(~V) = NaN;
dVhelp = squeeze(Vhelp(2:end,:,1) - Vhelp(1:end-1,:,1));
for i = 1:NPt-1
    meandVhelp = dVhelp(i,:);
    meandV(i) = abs(mean(meandVhelp(~isnan(meandVhelp))));
end

%dVdPt = squeeze(V(2:end,:,1) - V(1:end-1,:,1));
%meandV = abs(mean(dVdPt,2));

[~,IX] = sort(meandV,'descend');
b1 = min([IX(1) IX(2)]);    % lower Pt boundary
b2 = max([IX(1) IX(2)]);    % upper Pt boundary

% do initial regression

% set up regression vectors for 3 planes
regvec1 = zeros(b1*length(pii),2);
regvec1(:,1) = kron(ones(length(pii),1),Pt(1:b1)');
regvec1(:,2) = kron(pii',ones(b1,1));

regvec2 = zeros((b2-b1)*length(pii),2);
regvec2(:,1) = kron(ones(length(pii),1),Pt(b1+1:b2)');
regvec2(:,2) = kron(pii',ones(b2-b1,1));

regvec3 = zeros((length(Pt)-b2)*length(pii),2);
regvec3(:,1) = kron(ones(length(pii),1),Pt(b2+1:end)');
regvec3(:,2) = kron(pii',ones((length(Pt)-b2),1));

% initialize matrix to store regressoin coefficients
coefmat = zeros(T,3,3); % time, plane, param
coefmat(end,:,:) = [3 -1 -1; 3 -1 -1; 3 -1 -1];

% calculate regression parameters for each timestep
for t = 1:T-1
    Vdum = squeeze(Vhelp(:,:,t));
    V1dum = Vdum(1:b1,:);
    V2dum = Vdum(b1+1:b2,:);
    V3dum = Vdum(b2+1:end,:);
    V1 = [regvec1 V1dum(:)];
    V2 = [regvec2 V2dum(:)];
    V3 = [regvec3 V3dum(:)];
    
    % exclude points that haven't been visited
    V1(any(isnan(V1),2),:) = [];
    V2(any(isnan(V2),2),:) = [];
    V3(any(isnan(V3),2),:) = [];
    
    % do regression of 3 planes
    V1fit = fit(V1(:,1:2), V1(:,3),'poly11');
    V2fit = fit(V2(:,1:2), V2(:,3),'poly11');
    V3fit = fit(V3(:,1:2) ,V3(:,3),'poly11');
    
    coefmat(t,1,:) = coeffvalues(V1fit);
    coefmat(t,2,:) = coeffvalues(V2fit);
    coefmat(t,3,:) = coeffvalues(V3fit);
end
coefmatold = coefmat;

% plot function values from lookup table
figure
surf(Pt,pii,squeeze(Vhelp(:,:,1))')
view(142.5, 30)
xlabel('Concentration')
ylabel('Probability')
zlabel('Ob fun value')
title('Lookup table values')
saveas(gcf,'../../../Desktop/lakeproblem/lookup0','epsc')

%% ADP stage

% Within each plane, the objective function is linear in the decision
% variable (loading rate). Therefore the optimal loading rate will either
% be 0, 1, or such that it places the expected concentration (or the 5th or
% 95th percentile concentration, which are also used to calculate the
% expected value of the objective function at t+1) on a plane boundary

% JUST GIVE IT THE CORRECT VALUES TO TEST CODE
%b1 = 9;
%b2 = 29;

% plane boundaries to test
testbnd = Pt([b1 b1+1 b2 b2+1]);


% number of ADP iterations
M = 30000;
for m = 1:M
    m
    % start somewhere random  
    randdum = randperm(NPt);
    S = Pt(randdum(1));
    randdum2 = randperm(Npii);
    P = pii(randdum2(1));
    
    % sample path of concentrations, probabilities, and objective function
    % values for each timestep
    newVs = zeros(3,T-1);
    for t = 1:T-1
        newVs(1,t) = S;
        newVs(2,t) = P;
        
        % find lt points on plane boundaries to test
        ltbndhelp = testbnd - gmma*S - b;
        ltbndhelp2 = [pct5 + (S>Pcrit1)*r;
            (S>Pcrit1)*r;
            (S>Pcrit1)*r + pct95;
            (S>Pcrit2)*r + pct5;
            (S>Pcrit2)*r;
            (S>Pcrit2)*r + pct95];
        
        % generate matrix of loading rates to test, columns = which testbnd
        % points, rows = relevant outcomes for EV calculation (mean, 5th,
        % and 95th percentiles under each model
        ltbnd = kron(ones(6,1),ltbndhelp) - kron(ones(1,4),ltbndhelp2);
        
        % get rid of loading rates outside domain
        ltbnd(ltbnd<0) = NaN;
        ltbnd(ltbnd>lt(end)) = NaN;
        ltbnd = ltbnd(:);
        ltbnd = ltbnd(~isnan(ltbnd));
        
        % also test loading rates of 0 and 1
        ltbnd = [ltbnd; 0; lt(end)];
        
        % get rid of duplicates
        [~,index] = unique(ltbnd,'first');
        ltbnd = ltbnd(sort(index));
        
        % evaluate functions for each loading rate
        Vdum = zeros(1,length(ltbnd));
        for i = 1:length(ltbnd)
            % current cost function
            U = bta*ltbnd(i) - S^2;
            
            % find mean, 5th, and 95th percentile concentration values for
            % next timestep
            m1 = gmma*S + b + ltbnd(i) + (S>Pcrit1)*r;
            p5_1 = m1+pct5;
            p95_1 = m1+pct95;
            m2 = gmma*S + b + ltbnd(i) + (S>Pcrit2)*r;
            p5_2 = m2+pct5;
            p95_2 = m2+pct95;
            pts = [p5_1 m1 p95_1 p5_2 m2 p95_2]';
            
            % make a matrix with appropriate regression coefficients for
            % t+1 concentrations
            coefmat2 = kron((pts<=Pt(b1)),squeeze(coefmat(t+1,1,:))')...
                + kron((pts<=Pt(b2))&(pts>Pt(b1)),squeeze(coefmat(t+1,2,:))')...
                + kron((pts>Pt(b2)),squeeze(coefmat(t+1,3,:))');
            
            % likelihood functions and Bayesian updating
            Lt1 = exp(-(pts - pts(2)).^2/(2*sgma^2));
            Lt2 = exp(-(pts - pts(5)).^2/(2*sgma^2));
            piplus = P*Lt1./(P*Lt1 + (1-P)*Lt2);
            
            % variable matrix
            varmat = [ones(1,6); pts'; piplus'];
            
            % calculate value function for t+1, then EV
            Vprep = diag(coefmat2*varmat);
            EVmult = [P*[.185 .63 .185] (1-P)*[.185 .63 .185]];
            Vdum(i) = U + dlta*EVmult*Vprep;
        end
        
        % sometimes use a random loading rate instead of the optimal one
        if rand <= p
            ltdum = rand;
            U = bta*ltdum - S^2;

            m1 = gmma*S + b + ltdum + (S>Pcrit1)*r;
            p5_1 = m1+pct5;
            p95_1 = m1+pct95;
            m2 = gmma*S + b + ltdum + (S>Pcrit2)*r;
            p5_2 = m2+pct5;
            p95_2 = m2+pct95;
            pts = [p5_1 m1 p95_1 p5_2 m2 p95_2]';
            coefmat2 = kron((pts<=Pt(b1)),squeeze(coefmat(t+1,1,:))')...
                + kron((pts<=Pt(b2))&(pts>Pt(b1)),squeeze(coefmat(t+1,2,:))')...
                + kron((pts>Pt(b2)),squeeze(coefmat(t+1,3,:))');
            
            % likelihood functions and Bayesian updating
            Lt1 = exp(-(pts - pts(2)).^2/(2*sgma^2));
            Lt2 = exp(-(pts - pts(5)).^2/(2*sgma^2));
            piplus = P*Lt1./(P*Lt1 + (1-P)*Lt2);
            
            % put together variables
            varmat = [ones(1,6); pts'; piplus'];
            
            % calculate value function for t+1 in preparation for EV
            Vprep = diag(coefmat2*varmat);
            EVmult = [P*[.185 .63 .185] (1-P)*[.185 .63 .185]];
            Vnew = U + dlta*EVmult*Vprep;
        else % just take optimal value
            ltdum = ltbnd(Vdum==max(Vdum));
            Vnew = max(Vdum);
        end
        
        % figure out which regression plane you're on and calculate error
        whichplane = (S<=Pt(b1)) + 2*((S>Pt(b1))&(S<=Pt(b2)))...
            + 3*(S>Pt(b2));
        error = Vnew - squeeze(coefmat(t,whichplane,:))'*[1; S; P];
        
        % calculate gradient
        grad = zeros(1,1,3);
        grad(:,:,:) = [-1; -S; -P];
        
        % choose step size
        alfa = 1/(m+N);
        alfamult = 10;  %experiment with changing its size
        alfa = alfa*alfamult;
        
        % update parameter
        coefmat(t,whichplane,:) = coefmat(t,whichplane,:) - alfa*error*grad;
        
        % store initial state, probability, loading rate, function value,
        % and coefficient estimates
        if t==1
            results.new(m,1) = S;
            results.new(m,2) = P;
            results.new(m,3) = ltdum;
            results.new(m,4) = Vnew;
            results.new(m,5) = error;
            
            results.coefupd(m,:,:) = coefmat(t,:,:);
        end

        % update concentration and probability for next tiemstep
        Sdum = gmma*S + b + ltdum + P*r*(S>Pcrit1) + (1-P)*r*(S>Pcrit2) + randn*sgma;        
        Lt1b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit1)*r))^2/(2*sgma^2));
        Lt2b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit2)*r))^2/(2*sgma^2));
        Pdum = P*Lt1b/(P*Lt1b + (1-P)*Lt2b);
        if Sdum < 0     % update concentration for next timestep
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = Sdum;
        end
        P = Pdum;    % update probability estimate
        
    end
end

%results.ltopt = ltopt;
results.V = V;
results.Pt = Pt;
results.lt = lt;
results.pii = pii;
results.coefmat = coefmat;
results.b1 = b1;
results.b2 = b2;
results.coefmatold = coefmatold;

%% diagnostic plots

%coefmat = results.coefmat;
%coefmat

% final regression planes
f1 = @(x,y) squeeze(coefmat(1,1,:))'*[1; x; y];
f2 = @(x,y) squeeze(coefmat(1,2,:))'*[1; x; y];
f3 = @(x,y) squeeze(coefmat(1,3,:))'*[1; x; y];
figure
hold on
ezmesh(f1,[0 Pt(b1) 0 1])
ezmesh(f2,[Pt(b1) Pt(b2) 0 1])
ezmesh(f3,[Pt(b2) 1 0 1])
xlim([0 1])
zlim([0 5])
xlabel('Concentration')
ylabel('Probability')
zlabel('Ob fun value')
title('Final value function')
view(142.5, 30)
%saveas(gcf,'../../../Desktop/lakeproblem/finalval0','epsc')

% initial regression planes (regressed from lookup table)
f1 = @(x,y) squeeze(coefmatold(1,1,:))'*[1; x; y];
f2 = @(x,y) squeeze(coefmatold(1,2,:))'*[1; x; y];
f3 = @(x,y) squeeze(coefmatold(1,3,:))'*[1; x; y];
figure
hold on
ezmesh(f1,[0 Pt(b1) 0 1])
ezmesh(f2,[Pt(b1) Pt(b2) 0 1])
ezmesh(f3,[Pt(b2) 1 0 1])
xlim([0 1])
zlim([0 5])
xlabel('Concentration')
ylabel('Probability')
zlabel('Ob fun value')
title('Initial value function')
view(142.5, 30)
%saveas(gcf,'../../../Desktop/lakeproblem/initval0','epsc')

% parameter estimates as a function of iteration
figure
for i = 1:3
    subplot(1,3,i)
    hold on
    plot(squeeze(results.coefupd(:,i,1)))
    plot(squeeze(results.coefupd(:,i,2)),'r')
    plot(squeeze(results.coefupd(:,i,3)),'g')
    legend({'\beta_0' '\beta_1' '\beta_2'},'Location','East')
    title(['Plane' num2str(i) 'parameters'])
end
%saveas(gcf,'../../../Desktop/lakeproblem/params0','epsc')

% optimal loading rate
figure
scatter3(results.new(:,1), results.new(:,2), results.new(:,3))
xlabel('Concentration')
ylabel('Probability')
zlabel('Optimal loading rate')
view(142.5,30)

end
        
        
        
        
        
        
        
        
        
        