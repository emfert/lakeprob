% This implements the lake problem as done in the Lempert paper, using a
% linear model. Unlike previous versions, it keeps track of the previous
% loading rate as a state.

function results = Bond_ADP_reg2()
% set up initial parameters
clear
Pcrit = [.2 .7];     % candidate eutrophication thresholds
B = .2;                 % decay rate of P concentration (B in lempert paper)
b = .1;                 % natural baseline loading
r = .25;                % P recycling parameter
dlta = 1/1.03;          % discount factor
alphaa = 1;             % relative marginal utility of loadings
sgma = .04;             % st dev of stochastic shock
bbeta = 10;             % eutrophic cost
phi = 10;               % emissions reduction cost constant

N = 100000;                % no. samples total, for initial data collection
p = 5;                  % probabilit it jumps to a random decision

pct5 = norminv(.05,0,sgma);
pct95 = norminv(.95,0,sgma);

NPt = 21;               % no. grid points for Pt (concentration)
Npii = 21;              % no. grid points for pii (probabilities)
Nlt = 41;              % no. grid points for P loadings

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
lt = linspace(0,.8,Nlt);
T = 20;                 % time span

%% sample points to fit initial regression

% initialize lookup table array
V = ones([Nlt NPt Npii*ones(1,length(Pcrit)-1) T]);
%for i = 1:NPt
%    V(i,:,end) = 3 - pii - Pt(i);      % find appropriate final condition
%end
% for i = 1:Nlt
%     for j = 1:NPt
%         for k = 1:Npii
%             V(i,j,k) = 3 - lt(i) - Pt(j) - pii(k);
%         end
%     end
% end

% initialize array to store optimal loadings
ltopt = zeros([Nlt NPt Npii*ones(1,length(Pcrit)-1) T]);

% generate lookup table for first N samples
tic
for n = 1:N
    n
    % initial concentration
    randdum = randperm(NPt);
    S = Pt(randdum(1));
    
    % initial previous loading = 0
    lt_prev = lt(1);
    
    % sample random probability distribution
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
            % calculate current utility
            U = alphaa*lt(k) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(k))*(lt_prev>lt(k));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each candidate
            % threshold
            pts = zeros(3,length(Pcrit));
            pts(2,:) = B*S + b + lt(k) + r*(S>Pcrit);
            pts(1,:) = pts(2,:) + pct5;
            pts(3,:) = pts(2,:) + pct95;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(3,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                Lt(:,:,i) = exp(-(pts - pts(2,i)).^2/(2*sgma^2));
            end
            
            % generate array that's no. test points for calculating EV for
            % each model X number of models to test X number of predictors
            piplus = zeros(3,length(Pcrit),length(Pcrit));
            for i = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,i,:))*P';
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,i,:) = (squeeze(Lt(:,i,:)).*kron(ones(3,1),P))./pi_help2;
            end
            
            % generate indices for lookup table independent variables (for
            % probability distribution)
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
            % extract V(t+1) from lookup table for each point necessary for
            % EV calculation
            for i = 1:3
                for j = 1:length(Pcrit)
                    pi_dum = [];
                    for jj = 1:length(Pcrit)-1
                        pi_dum = [pi_dum 'piplus_ind(i,j,' num2str(jj) '),'];
                    end
                    Vpts(i,j) = eval(['squeeze(V(lt==lt_prev,Pt==S,' pi_dum 't+1))']);
                end
            end

            % EV of V(t+1)
            E = [.185 .63 .185]*Vpts;            
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
        
        % put decisions and function values in lookup tables
        ltdum = lt(Vnt==Vdum);                
        P_ind_dum = [];
        for i = 1:length(Pcrit)-1
            P_ind_dum = [P_ind_dum 'P_ind(' num2str(i) '),'];
        end
        eval(['V(lt==lt_prev,Pt==S,' P_ind_dum 't) = Vnt;']);
        eval(['ltopt(lt==lt_prev,Pt==S,' P_ind_dum 't) = ltdum;']);
                
        % simulate concentration and probability distribution for next
        % time step, store current loading rate for next period calculation
        lt_prev = ltdum;
        
        p_log = zeros(length(Pcrit),1);
        for i = 1:length(Pcrit)
            p_log(i) = S>Pcrit(i);
        end        
        Sdum = B*S + b + ltdum + r*P*p_log + randn*sgma;
        
        Ltb = exp(-(Sdum - (B*S + b + ltdum + r*p_log)).^2/(2*sgma^2));        
        Pdum = P.*Ltb'/(P*Ltb);
        
        % make concentrations, probabilities compatible with lookup table
        % lattice
        if Sdum < 0
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
toc
%% or just load points from a workspace, if not sampling - run top cell, then...
% load BondADP10k
% V = results.V;

% find concentrations in lattice closest to threshold candidates
bnds = zeros(1,length(Pcrit));
for i = 1:length(Pcrit)
    bnds(i) = find(abs(Pcrit(i)-pii)==min(abs(Pcrit(i)-pii)));
end
bnds = [0 bnds NPt];

% set up regression matrices for each partitioning of state space
% (dependent on threshold candidates)
for i = 1:length(bnds)-1
    regvecdum = ones((bnds(i+1)-bnds(i))*Npii^(length(Pcrit)-1)*Nlt,length(Pcrit)+2);
    regvecdum(:,2) = kron(ones(Npii^(length(Pcrit)-1)*(bnds(i+1)-bnds(i)),1),lt');
    regvecdum(:,3) = kron(ones(Npii^(length(Pcrit)-1),1),kron(Pt(bnds(i)+1:bnds(i+1))',ones(Nlt,1)));
    for j = 4:length(Pcrit)+2
        regvecdum(:,j) = kron(ones(Npii^(length(Pcrit)-j+2),1), kron(pii',ones((bnds(i+1)-bnds(i))*Nlt*Npii^(j-4),1)));
    end
    assignin('base',['regvec' num2str(i)],regvecdum);
end

% initialize matrix to store regression coefficients
coefmat = zeros(T,3,length(Pcrit)+2); % time, plane, param
coefmat(end,:,:) = [3 -1 -1 -1*ones(1,length(Pcrit)-1); 3 -1 -1 -1*ones(1,length(Pcrit)-1); 3 -1 -1 -1*ones(1,length(Pcrit)-1)];

% do regression for each timestep
for t = 1:T-1
    Vdumhelp = [];
    for i = 1:length(Pcrit)-1
        Vdumhelp = [Vdumhelp ':,'];
    end
    Vdum = squeeze(eval(['V(:,:,' Vdumhelp 't)']));    % extract lookup table values for V    
    for i = 1:length(bnds)-1
        Vdum2 = Vdum(:,bnds(i)+1:bnds(i+1),:);
        Vdum3 = Vdum2(:);
        Vdum3(Vdum3==1) = NaN;
        coefmat(t,i,:) = regress(Vdum3,eval(['regvec' num2str(i)]));
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

M = 100;                % number of ADP iterations
bnds(1) = 1;

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
    
    % initial prior loading rate
    lt_prev = 0;      
    
    for t = 1:T-1
        
        % evaluate functions for each loading rate
        Vdum = zeros(1,Nlt);
        for i = 1:Nlt
            U = alphaa*lt(i) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(i))*(lt_prev>lt(i));
            
            % rows are 5th, mean, 95th percentile next-period
            % concentrations, columns correspond to each threshold
            pts = zeros(3,length(Pcrit));
            pts(2,:) = B*S + b + lt(i) + r*(S>Pcrit);
            pts(1,:) = pts(2,:) + pct5;
            pts(3,:) = pts(2,:) + pct95;
                        
            % Lt(:,:,i) is the likelihood of each point in pts given model
            % i
            Lt = zeros(3,length(Pcrit),length(Pcrit));
            for j = 1:length(Pcrit)
                Lt(:,:,j) = exp(-(pts - pts(2,j)).^2/(2*sgma^2));
            end
            
            % make array that's no. test points for calculating EV for
            % each model X number of models to test X number of predictors
            piplus = zeros(3,length(Pcrit),length(Pcrit));
            for j = 1:length(Pcrit)
                pi_help = squeeze(Lt(:,j,:))*P';
                pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
                piplus(:,j,:) = (squeeze(Lt(:,j,:)).*kron(ones(3,1),P))./pi_help2;
            end   
            
            % make a matrix with appropriate regression coefficients for
            % t+1 concentrations
            coefmat2 = zeros(3,length(Pcrit),length(Pcrit)+2);
            for j = 1:3
                for k = 1:length(Pcrit)
                    for jj = 1:length(bnds)-1
                        if (pts(j,k)>Pt(bnds(jj)))&&(pts(j,k)<=Pt(bnds(jj+1)))
                            coefmat2(j,k,:) = squeeze(coefmat(t+1,jj,:));
                        end
                    end
                end
            end
                        
            % calculate value function for t+1
            Vtp1 = zeros(size(pts));
            for j = 1:3
                for k = 1:length(Pcrit)
                    Vtp1(j,k) = squeeze(coefmat2(j,k,:))'*[1 lt_prev S P(1:end-1)]';
                end
            end
                        
            % calculate EV for t+1
            EVmult = zeros(3,length(Pcrit));
            for j = 1:length(Pcrit)
                EVmult(:,j) = P(j)*[.185 .63 .185]';
            end
            
            Vdum(i) = U + dlta*sum(sum(EVmult.*Vtp1));
        end
        
        % sometimes use a random loading rate instead of the optimal one
        if rand <= p
            randdum2 = randperm(Nlt);
            ltdum = lt(randdum2(1));
            Vnew = Vdum(randdum2(1));
        else % just take optimal value
            ltdum = ltbnd(Vdum==max(Vdum));
            Vnew = max(Vdum);
        end
        
        % figure out which regression plane you're on and calculate error
        for j = 1:length(bnds)-1
            if (S>Pt(bnds(j)))&&(S<=Pt(bnds(j+1)))
                whichplane = j;
            end
        end
        
        error = Vnew - squeeze(coefmat(t,whichplane,:))'*[1; lt_prev; S; P(1:end-1)'];
        
        % calculate gradient
        grad = zeros(1,1,2+length(Pcrit));
        grad(:,:,:) = [-1; lt_prev; -S; -P(1:end-1)'];

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
        p_log = zeros(length(Pcrit),1);
        for i = 1:length(Pcrit)
            p_log(i) = S>Pcrit(i);
        end        
        Sdum = B*S + b + ltdum + r*P*p_log + randn*sgma;
        
        % update probability distribution given simulated concentration
        Ltb = exp(-(Sdum - (B*S + b + ltdum + r*p_log)).^2/(2*sgma^2));        
        P = P.*Ltb'/(P*Ltb);
        
        % no nonphysical concentrations
        if Sdum < 0     % update concentration for next timestep
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
        end        
    end
end

results.V = V;
results.Pt = Pt;
results.lt = lt;
results.ltopt = ltopt;
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

% % seed rng
% 
% P = [.6 .4];        % initial guess for prob dist
% X0 = .787;              % initial state
% Xcrit = .6;             % actual Xcrit value
% NN = 1000;              % number of sample paths to simulate
% T2 = 100;               % how far out in time to simulate them
% 
% % get coefficient matrix that's 
% coefmatsim = squeeze(coefmat(5,:,:));
% 
% % initialize concentration and loading storage matrices
% X = zeros(NN,T2);
% L = zeros(NN,T2);
% deltaL = zeros(NN,T2);
% 
% X(:,1) = X0;
% 
% for i = 1:NN
%     lt_prev = 0;
%     for t = 2:T2
%         % calculate optimal loading rate
%         Vdum = zeros(Nlt,1);
%         for j = 1:Nlt
%             % calculate current utility
%             U = alphaa*lt(j) - bbeta*(X(i,t)>Pcrit)*P' - phi*(lt_prev-lt(j))*(lt_prev>lt(j));
%             
%             % FIND WHICHPLANE
%             Vdum(j) = 
%         end
%     end
% end
end
        
        
        
        
        
        
        
        
        
        