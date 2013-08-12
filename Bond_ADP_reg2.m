% This implements the lake problem as done in the Lempert paper, using a
% linear model. Unlike previous versions, it keeps track of the previous
% loading rate as a state.

function results = Bond_ADP_reg2()
% set up initial parameters
clear
Pcrit = [.3 .5 .8];
B = .2;              % decay rate of P concentration (B in lempert paper)
b = .1;                 % natural baseline loading
r = .25;                % P recycling parameter
dlta = 1/1.03;          % discount factor
alphaa = 1;             % relative marginal utility of loadings
sgma = .04;             % st dev of stochastic shock
bbeta = 10;             % eutrophic cost
phi = 10;               % emissions reduction cost constant

N = 100;             % no. samples total, for initial data collection
p = 5;                  % probabilit it jumps to a random decision

pct5 = norminv(.05,0,sgma);
pct95 = norminv(.95,0,sgma);

NPt = 20;%41;               % no. grid points for Pt (concentration)
Npii = 20;%41;              % no. grid points for pii (probabilities)
Nlt = 20;%161;              % no. grid points for P loadings

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
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
                % HAVE TO HARD CODE THIS - TRIED NOT TO
                for j = 1:length(Pcrit)
                    pi_dum = [];
                    for jj = 1:length(Pcrit)-1
                        pi_dum = [pi_dum 'piplus_ind(i,j,' num2str(jj) '),'];
                    end
                    %Vpts(i,j) = squeeze(V(lt==lt_prev,Pt==S,piplus_ind(i,j,1),t+1));
                    Vpts(i,j) = eval(['squeeze(V(lt==lt_prev,Pt==S,' pi_dum 't+1))']);
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
        
        % HARD CODE - TEST!!       
        % V(lt==lt_prev,Pt==S,P_ind(1),t) = Vnt;
        % ltopt(lt==lt_prev,Pt==S,P_ind(1),t) = ltdum;        
        P_ind_dum = [];
        for i = 1:length(Pcrit)-1
            P_ind_dum = [P_ind_dum 'P_ind(' num2str(i) '),'];
        end
        eval(['V(lt==lt_prev,Pt==S,' P_ind_dum 't) = Vnt']);
        eval(['ltopt(lt==lt_prev,Pt==S,' P_ind_dum 't) = ltdum']);
                
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

% b1 = bnds(1);
% b2 = bnds(2);

% HARD CODE THIS INITIALLY FOR NUMBER OF PLANES
% set up regression vectors for 3 planes
% !!!! STILL NEED TO FIX REGRESSION VECTORS FOR MORE PROB DIMENSIONS
bnds = [0 bnds NPt];
for i = 1:length(bnds)-1
%     regvecdum = ones((bnds(i+1)-bnds(i))*Npii*Nlt,length(Pcrit)+2);
%     regvecdum(:,2) = kron(ones(Npii*(bnds(i+1)-bnds(i)),1),lt');
%     regvecdum(:,3) = kron(ones(Npii,1),kron(Pt(bnds(i)+1:bnds(i+1))',ones(Nlt,1)));
%     regvecdum(:,4:end) = kron(pii',ones((bnds(i+1)-bnds(i))*Nlt,length(Pcrit)-1));
    regvecdum = ones((bnds(i+1)-bnds(i))*Npii^(length(Pcrit)-1)*Nlt,length(Pcrit)+2);
    regvecdum(:,2) = kron(ones(Npii^(length(Pcrit)-1)*(bnds(i+1)-bnds(i)),1),lt');
    regvecdum(:,3) = kron(ones(Npii^(length(Pcrit)-1),1),kron(Pt(bnds(i)+1:bnds(i+1))',ones(Nlt,1)));
    for j = 4:length(Pcrit)+2
        regvecdum(:,j) = kron(ones(Npii^(length(Pcrit)-j+2),1), kron(pii',ones((bnds(i+1)-bnds(i))*Nlt*Npii^(j-4),1)));
    end
    assignin('base',['regvec' num2str(i)],regvecdum);
end

% regvec1 = ones(b1*Npii*Nlt,4);
% regvec1(:,2) = kron(ones(Npii*b1,1),lt');
% regvec1(:,3) = kron(ones(Npii,1),kron(Pt(1:b1)',ones(Nlt,1)));
% regvec1(:,4) = kron(pii',ones(b1*Nlt,1));
% 
% regvec2 = ones((b2-b1)*Npii*Nlt,4);
% regvec2(:,2) = kron(ones(Npii*(b2-b1),1),lt');
% regvec2(:,3) = kron(ones(Npii,1),kron(Pt(b1+1:b2)',ones(Nlt,1)));
% regvec2(:,4) = kron(pii',ones((b2-b1)*Nlt,1));
% 
% regvec3 = ones((length(Pt)-b2)*Npii*Nlt,4);
% regvec3(:,2) = kron(ones(Npii*(NPt-b2),1),lt');
% regvec3(:,3) = kron(ones(Npii,1),kron(Pt(b2+1:end)',ones(Nlt,1)));
% regvec3(:,4) = kron(pii',ones((length(Pt)-b2)*Nlt,1));

% initialize matrix to store regressoin coefficients
coefmat = zeros(T,3,length(Pcrit)+2); % time, plane, param
coefmat(end,:,:) = [3 -1 -1 -1*ones(1,length(Pcrit)-1); 3 -1 -1 -1*ones(1,length(Pcrit)-1); 3 -1 -1 -1*ones(1,length(Pcrit)-1)];

% calculate regression parameters for each timestep
for t = 1:T-1
    Vdumhelp = [];
    for i = 1:length(Pcrit)-1
        Vdumhelp = [Vdumhelp ':,'];
    end
    Vdum = squeeze(eval(['V(:,:,' Vdumhelp 't)']));    % extract lookup table values for V
%     V1dum = Vdum(:,1:b1,:);
%     V2dum = Vdum(:,b1+1:b2,:);
%     V3dum = Vdum(:,b2+1:end,:);
%     V1 = V1dum(:);
%     V2 = V2dum(:);
%     V3 = V3dum(:);
%     
%     % exclude points that haven't been visited
%     V1(V1==1) = NaN;
%     V2(V2==1) = NaN;
%     V3(V3==1) = NaN;
    
    for i = 1:length(bnds)-1
        Vdum2 = Vdum(:,bnds(i)+1:bnds(i+1),:);
        Vdum3 = Vdum2(:);
        Vdum3(Vdum3==1) = NaN;
        coefmat(t,i,:) = regress(Vdum3,eval(['regvec' num2str(i)]));
    end
    
    % do regression of 3 planes
%     b1h = regress(V1,regvec1);
%     b2h = regress(V2,regvec2);
%     b3h = regress(V3,regvec3);
%     
%     coefmat(t,1,:) = b1h;
%     coefmat(t,2,:) = b2h;
%     coefmat(t,3,:) = b3h;
    
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
    S = rand*Pt(end);   % initial random concentration
    
    dif = 0;            % initial random probability distribution
    P = zeros(1,length(Pcrit));
    for i = 1:length(Pcrit)-1
        P(i) = rand*(1-dif);
        dif = dif+P(i);
    end
    P(end) = 1-dif;
    
    lt_prev = 0;        % initial previous loading rate
    
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
%                     coefmat2(j,k,:) = (pts(j,k)<=Pt(b1))*squeeze(coefmat(t+1,1,:))...
%                         + (pts(j,k)<=Pt(b2))&(pts(j,k)>Pt(b1))*squeeze(coefmat(t+1,2,:))...
%                         + (pts(j,k)>Pt(b2))*squeeze(coefmat(t+1,3,:));
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
        %wp = [0 Pcrit];
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
        alfamult = 100;  %experiment with changing its size
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
%results.b1 = b1;
%results.b2 = b2;
results.bnds = bnds;
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
        
        
        
        
        
        
        
        
        
        