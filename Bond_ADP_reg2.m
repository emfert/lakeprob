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

N = 100000;             % no. samples total, for initial data collection
p = 5;                  % probabilit it jumps to a random decision

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
            pts(2,:) = B*S + b + lt(k) + r*(S>Pcrit);
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
% load BondADP10k
% V = results.V;
% Pt = results.Pt;
% pii = results.pii;

bnds = zeros(1,length(Pcrit));
for i = 1:length(Pcrit)
    bnds(i) = find(abs(Pcrit(i)-pii)==min(abs(Pcrit(i)-pii)));
end

b1 = bnds(1);
b2 = bnds(2);

% do initial regression RESUME HERE!!!
% HARD CODE THIS INITIALLY FOR NUMBER OF PLANES
% set up regression vectors for 3 planes
regvec1 = ones(b1*length(pii)*Nlt,4);
regvec1(:,2) = kron(ones(length(pii)*b1,1),lt');
regvec1(:,3) = kron(ones(length(pii),1),kron(Pt(1:b1)',ones(Nlt,1)));
regvec1(:,4) = kron(pii',ones(b1*Nlt,1));

regvec2 = ones((b2-b1)*length(pii)*Nlt,4);
regvec2(:,2) = kron(ones(length(pii)*(b2-b1),1),lt');
regvec2(:,3) = kron(ones(length(pii),1),kron(Pt(b1+1:b2)',ones(Nlt,1)));
regvec2(:,4) = kron(pii',ones((b2-b1)*Nlt,1));

regvec3 = ones((length(Pt)-b2)*length(pii)*Nlt,4);
regvec3(:,2) = kron(ones(length(pii)*(NPt-b2),1),lt');
regvec3(:,3) = kron(ones(length(pii),1),kron(Pt(b2+1:end)',ones(Nlt,1)));
regvec3(:,4) = kron(pii',ones((length(Pt)-b2)*Nlt,1));

% initialize matrix to store regressoin coefficients
coefmat = zeros(T,3,4); % time, plane, param
coefmat(end,:,:) = [3 -1 -1 -1; 3 -1 -1 -1; 3 -1 -1 -1];

% calculate regression parameters for each timestep
for t = 1:T-1
    Vdum = squeeze(V(:,:,:,t));
    V1dum = Vdum(:,1:b1,:);
    V2dum = Vdum(:,b1+1:b2,:);
    V3dum = Vdum(:,b2+1:end,:);
    V1 = V1dum(:);
    V2 = V2dum(:);
    V3 = V3dum(:);
    
    % exclude points that haven't been visited
    %V1(any(isnan(V1),2),:) = [];
    %V2(any(isnan(V2),2),:) = [];
    %V3(any(isnan(V3),2),:) = [];
    
    V1(V1==1) = NaN;
    V2(V2==1) = NaN;
    V3(V3==1) = NaN;
    
    % do regression of 3 planes
    %V1fit = fit(V1(:,1:2), V1(:,3),'poly11');
    %V2fit = fit(V2(:,1:2), V2(:,3),'poly11');
    %V3fit = fit(V3(:,1:2) ,V3(:,3),'poly11');
    
    %coefmat(t,1,:) = coeffvalues(V1fit);
    %coefmat(t,2,:) = coeffvalues(V2fit);
    %coefmat(t,3,:) = coeffvalues(V3fit);
    b1h = regress(V1,regvec1);
    b2h = regress(V2,regvec2);
    b3h = regress(V3,regvec3);
    
    coefmat(t,1,:) = b1h;
    coefmat(t,2,:) = b2h;
    coefmat(t,3,:) = b3h;
    
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

% number of ADP iterations
M = 100;
for m = 1:M
    m
    % start somewhere random  
    S = rand*Pt(end);
    
    dif = 0;
    P = zeros(1,length(Pcrit));
    for i = 1:length(Pcrit)-1
        P(i) = rand*(1-dif);
        dif = dif+P(i);
    end
    P(end) = 1-dif;
    
    lt_prev = 0;
    
    % sample path of concentrations, probabilities, and objective function
    % values for each timestep
    
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
            
            % make matrix that's no. test points for calculating EV for
            % each model : number of models to test: number of predictors
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
                    coefmat2(j,k,:) = (pts(j,k)<=Pt(b1))*squeeze(coefmat(t+1,1,:))...
                        + (pts(j,k)<=Pt(b2))&(pts(j,k)>Pt(b1))*squeeze(coefmat(t+1,2,:))...
                        + (pts(j,k)>Pt(b2))*squeeze(coefmat(t+1,3,:));
                end
            end
                        
            % variable matrix
            Vtp1 = zeros(size(pts));
            for j = 1:3
                for k = 1:length(Pcrit)
                    Vtp1(j,k) = squeeze(coefmat2(j,k,:))'*[1 lt_prev S P(1:end-1)]';
                end
            end
            
            %varmat = [ones(1,length(Pcrit)*3); pts(:); piplus(:)];
            
            % calculate value function for t+1, then EV
            %Vprep = diag(coefmat2*varmat);
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
            
%             ltdum = rand;
%             U = alphaa*lt(k) - bbeta*(S>Pcrit)*P' - phi*(lt_prev-lt(k))*(lt_prev>lt(k));
%             
%             % rows are 5th, mean, 95th percentile next-period
%             % concentrations, columns correspond to each threshold
%             pts = zeros(3,length(Pcrit));
%             pts(2,:) = B*S + b + lt(k) + (S>Pcrit);
%             pts(1,:) = pts(2,:) + pct5;
%             pts(3,:) = pts(2,:) + pct95;
%                         
%             % Lt(:,:,i) is the likelihood of each point in pts given model
%             % i
%             Lt = zeros(3,length(Pcrit),length(Pcrit));
%             for j = 1:length(Pcrit)
%                 Lt(:,:,j) = exp(-(pts - pts(2,j)).^2/(2*sgma^2));
%             end
%             
%             % make matrix that's no. test points for calculating EV for
%             % each model : number of models to test: number of predictors
%             piplus = zeros(3,length(Pcrit),length(Pcrit));
%             for j = 1:length(Pcrit)
%                 pi_help = squeeze(Lt(:,j,:))*P';
%                 pi_help2 = kron(ones(1,length(Pcrit)),pi_help);
%                 piplus(:,j,:) = (squeeze(Lt(:,j,:)).*kron(ones(3,1),P))./pi_help2;
%             end 
%             
%             coefmat2 = kron((pts<=Pt(b1)),squeeze(coefmat(t+1,1,:))')...
%                 + kron((pts<=Pt(b2))&(pts>Pt(b1)),squeeze(coefmat(t+1,2,:))')...
%                 + kron((pts>Pt(b2)),squeeze(coefmat(t+1,3,:))');
%             
%             % put together variables
%             Vtp1 = [ones(1,6); pts'; piplus'];
%             
%             % calculate value function for t+1 in preparation for EV
%             Vprep = diag(coefmat2*Vtp1);
%             EVmult = [P*[.185 .63 .185] (1-P)*[.185 .63 .185]];
%             Vnew = U + dlta*EVmult*Vprep;
        else % just take optimal value
            ltdum = ltbnd(Vdum==max(Vdum));
            Vnew = max(Vdum);
        end
        
        % figure out which regression plane you're on and calculate error
        % whichplane = (S<=Pt(b1)) + 2*((S>Pt(b1))&(S<=Pt(b2)))...
        %     + 3*(S>Pt(b2));
        wp = [0 Pcrit];
        for j = 1:length(Pcrit)
            if (wp(j)<=S)&&(wp(j+1)>S)
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
        
        % update parameter
        coefmat(t,whichplane,:) = coefmat(t,whichplane,:) - alfa*error*grad;
        
        % store initial state, probability, loading rate, function value,
        % and coefficient estimates
        if t==5
            results.new(m,1) = Vnew;
            results.new(m,2) = error;
            results.new(m,3) = lt_prev;
            results.new(m,4) = ltdum;
            results.new(m,5) = S;
            results.new(m,6:6+length(Pcrit)-1) = P';
            
            results.coefupd(m,:,:) = coefmat(t,:,:);
        end

%         % update concentration and probability for next tiemstep
%         Sdum = B*S + b + ltdum + P*r*(S>Pcrit1) + (1-P)*r*(S>Pcrit2) + randn*sgma;        
%         Lt1b = exp(-(Sdum - (B*S + b + ltdum + (S>Pcrit1)*r))^2/(2*sgma^2));
%         Lt2b = exp(-(Sdum - (B*S + b + ltdum + (S>Pcrit2)*r))^2/(2*sgma^2));
%         Pdum = P*Lt1b/(P*Lt1b + (1-P)*Lt2b);
%         if Sdum < 0     % update concentration for next timestep
%             S = 0;
%         elseif Sdum > 1
%             S = 1;
%         else
%             S = Sdum;
%         end
%         P = Pdum;    % update probability estimate
        
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
        
        
        
        
        
        
        
        
        
        