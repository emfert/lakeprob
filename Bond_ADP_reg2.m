% This implements the lake problem as done in the Lempert paper, using a
% linear model. Unlike previous versions, it keeps track of the previous
% loading rate as a state.

function results = Bond_ADP_reg2()
% set up initial parameters
clear
%Pcrit1 = .2; % or .7    % critical threshold
%Pcrit2 = .7;
Pcrit = [.2 .7];
B = .2;              % decay rate of P concentration (B in lempert paper)
b = .1;                 % natural baseline loading
r = .25;                % P recycling parameter
dlta = 1/1.03;          % discount factor
alphaa = 1;             % relative marginal utility of loadings
sgma = .04;             % st dev of stochastic shock
bbeta = 10;             % eutrophic cost
phi = 10;               % emissions reduction cost constant

N = 100000; %NOT ENOUGH               % no. samples total, for initial data collection
p = 5;                % probabilit it jumps to a random decision

pct5 = norminv(.05,0,sgma);
pct95 = norminv(.95,0,sgma);

NPt = 20;%41;               % no. grid points for Pt (concentration)
Npii = 20;%41;              % no. grid points for pii (probabilities)
Nlt = 20;%161;              % no. grid points for P loadings
%Hn = 16;                % Hermite nodes and weights
%eps = .001;             % Value function error tolerance

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
%pii = repmat(pii_help,length(Pcrit),1);
lt = linspace(0,.8,Nlt);
T = 10;                 % time span

%% sample points to fit initial regression

V = ones([Nlt NPt Npii*ones(1,length(Pcrit)-1) T]);
%for i = 1:NPt
%    V(i,:,end) = 3 - pii - Pt(i);      % find appropriate final condition
%end
% HARD CODE IN PROBABILITY DIMENSIONALITY
% for i = 1:Nlt
%     for j = 1:NPt
%         for k = 1:Npii
%             V(i,j,k) = 3 - lt(i) - Pt(j) - pii(k);
%         end
%     end
% end

ltopt = zeros([Nlt NPt Npii*ones(1,length(Pcrit)-1) T]);

% generate lookup table for first N samples
for n = 1:N
    n
    % initial state variables
    randdum = randperm(NPt);
    S = Pt(randdum(1));
    
    %rdum2 = randperm(Nlt);
    %lt_prev = lt(rdum2(1));
    lt_prev = lt(1);
    
    P_help = randperm(length(Pcrit)-1);
    dif = 0;
    P = zeros(1,length(Pcrit));
    P_ind = zeros(1,length(Pcrit));    
    for i = 1:length(P_help)
        P_raw = rand*(1-dif);
        P_ind(i) = find(abs(pii-P_raw)==min(abs(pii-P_raw)));
        P_help2 = pii(P_ind(i));
        P(P_help(i)) = P_help2;
        dif = dif+P(P_help(i));
    end
    P(end) = 1-dif;
    P_ind(end) = find(abs(P(end)-pii)==min(abs(P(end)-pii)));
        
    for t = 1:T-1
        Vdum = zeros(1,Nlt);
        for k = 1:Nlt               % iterate through control space                        
            % need to do EV with current prob dist over Pcrits, this won't
            % run as is
            U = alphaa*lt(k) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(k))*(lt_prev>lt(k));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each threshold
            pts = zeros(3,length(Pcrit));
            pts(2,:) = B*S + b + lt(k) + (S>Pcrit);
            pts(1,:) = pts(2,:) + pct5;
            pts(3,:) = pts(2,:) + pct95;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(3,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                Lt(:,:,i) = exp(-(pts - pts(2,i)).^2/(2*sgma^2));
            end
            
            % make matrix that's no. test points for calculating EV for
            % each model : number of models to test: number of predictors
            piplus = zeros(3,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,i,:))*P';
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,i,:) = (squeeze(Lt(:,i,:)).*kron(ones(3,1),P))./pi_help2;
            end
            
            % make matrix of lookup-table indices
            piplus_ind_help = zeros(3,length(Pcrit),length(Pcrit),Npii);
            Vpts = zeros(3,length(Pcrit));
            for i = 1:3
                for j = 1:length(Pcrit)
                    for jj = 1:length(Pcrit)
                        piplus_ind_help(i,j,jj,:) = piplus(i,j,jj) - pii';
                    end
                end
            end
            [~, piplus_ind] = min(abs(piplus_ind_help),[],4);
            for i = 1:3
                % HAVE TO HARD CODE THIS
                for j = 1:length(Pcrit)
                    %Vpts(i,j) = squeeze(V(Pt==S,piplus_ind(i,j,1),piplus_ind(i,j,2),t));
                    Vpts(i,j) = squeeze(V(lt==lt_prev,Pt==S,piplus_ind(i,j,1),t+1));
                end
            end

            % expected value under each of Pcrit models
            E = [.185 .63 .185]*Vpts;
            
            Vdum(k) = U + dlta*(E*P');
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
        
        % HARD CODE
        %V(Pt==S,P_ind(1),P_ind(2),t) = Vnt;
        %ltopt(Pt==S,P_ind(1),P_ind(2),t) = ltdum;        
        V(lt==lt_prev,Pt==S,P_ind(1),t) = Vnt;
        ltopt(lt==lt_prev,Pt==S,P_ind(1),t) = ltdum;
                
        % calculate (random) state and probability estimate for next
        % timestep
        
        lt_prev = ltdum;
        
        p_log = zeros(length(Pcrit),1);
        for i = 1:length(Pcrit)
            p_log(i) = S>Pcrit(i);
        end
        
        Sdum = B*S + b + ltdum + r*P*p_log + randn*sgma;
        
        Ltb = exp(-(Sdum - (B*S + b + ltdum + r*p_log)).^2/(2*sgma^2));
        
        Pdum = P.*Ltb'/(P*Ltb);
        if Sdum < 0     % update concentration for next timestep
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = Pt(abs(Sdum-Pt)==min(abs(Sdum-Pt)));
        end
        for i = 1:length(Pcrit)
            P(i) = pii(abs(Pdum(i)-pii)==min(abs(Pdum(i)-pii)));
        end
    end
end

%% or just load points from a workspace, if not sampling
load BondADP10k
V = results.V;
Pt = results.Pt;
pii = results.pii;

bnds = zeros(1,length(Pcrit));
for i = 1:length(Pcrit)
    bnds(i) = find(abs(Pcrit(i)-pii)==min(abs(Pcrit(i)-pii)));
end

% do initial regression RESUME HERE!!!

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
%saveas(gcf,'../../../Desktop/lakeproblem/lookup0','epsc')

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
        ltbndhelp = testbnd - B*S - b;
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
            U = alphaa*ltbnd(i) - S^2;
            
            % find mean, 5th, and 95th percentile concentration values for
            % next timestep
            m1 = B*S + b + ltbnd(i) + (S>Pcrit1)*r;
            p5_1 = m1+pct5;
            p95_1 = m1+pct95;
            m2 = B*S + b + ltbnd(i) + (S>Pcrit2)*r;
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
            U = alphaa*ltdum - S^2;

            m1 = B*S + b + ltdum + (S>Pcrit1)*r;
            p5_1 = m1+pct5;
            p95_1 = m1+pct95;
            m2 = B*S + b + ltdum + (S>Pcrit2)*r;
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
        Sdum = B*S + b + ltdum + P*r*(S>Pcrit1) + (1-P)*r*(S>Pcrit2) + randn*sgma;        
        Lt1b = exp(-(Sdum - (B*S + b + ltdum + (S>Pcrit1)*r))^2/(2*sgma^2));
        Lt2b = exp(-(Sdum - (B*S + b + ltdum + (S>Pcrit2)*r))^2/(2*sgma^2));
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
        
        
        
        
        
        
        
        
        
        