% this is a crack at using ADP to implement the lake problem from the Bond
% paper
% unlike Bond_ADP, it jumps around for the first Nr iterations, it doesn't
% just go straight to the optimal stuff
% it implements a function approximation for V rather than doing it point
% by point

function results = Bond_ADP_reg()

Pcrit1 = .2; % or .7    % critical threshold
Pcrit2 = .7;
gmma = .1;              % decay rate of P concentration
b = .02;                % natural baseline loading
r = .2;                 % P recycling parameter
dlta = .99;             % discount factor
bta = 1.5;              % relative marginal utility of loadings
sgma = .141421;         % st dev of stochastic shock
N = 10000;               % no. samples total
Nr = 500;               % no. samples to take random decision

pct5 = norminv(.05,0,sgma);
pct95 = norminv(.95,0,sgma);

NPt = 41;               % no. grid points for Pt
Npii = 41;              % no. grid points for pii
Nlt = 161;              % no. grid points for P loadings
Hn = 16;                % Hermite nodes and weights
eps = .001;             % Value function error tolerance

Pt = linspace(0,1,NPt);
pii = linspace(0,1,Npii);
lt = linspace(0,.8,Nlt);
T = 10;                 % time span

count = 1;              % set iteration counter

%Vold = zeros(NPt,Npii);
%V = zeros(NPt,Npii,T);
%for i = 1:NPt
%    V(i,:,end) = .5*pii-Pt(i);      % find appropriate final condition
%end

V = zeros(3,T);
V(:,T) = [0; -1; .5];               % regression parameters for value fcn

%ltopt = zeros(NPt,Npii,T);
%ltold = ones(NPt, Npii,T);
ltopt = NaN(3,T-1);                 % optimal P loading

%[X,Y] = meshgrid(pii,Pt');

%create sample path
%omga = randn(1,T+1);

U = @(x,y) bta*x - y^2;             % current utility of x flow, y stock

% means and 5th/95th percentiles as functions of flow x and stock y
mean1 = @(x,y) gmma*y + b + x + (y>Pcrit1)*r;
mean2 = @(x,y) gmma*y + b + x + (y>Pcrit2)*r;

for n = 1:N
    % initial state variables - random concentration and probability
    randdum = randperm(NPt);
    S = Pt(randdum(1));
    randdum2 = randperm(Npii);
    P = pii(randdum2(1));
    
    for t = 1:T-1
        %Vdum = zeros(1,Nlt);
        
        % define V as a function of P loading
        pts = @(x) [mean1(x,S)+pct5 mean1(x,S) mean1(x,S)+pct95 mean2(x,S)+pct5 mean2(x,S) mean2(x,S)+pct95];
        Lt1 = @(x) exp(-(pts(x) - mean1(x,S)).^2/(2*sgma^2));
        Lt2 = @(x) exp(-(pts(x) - mean2(x,S)).^2/(2*sgma^2));
        piplus = @(x) P*Lt1(x)./(P*Lt1(x) + (1-P)*Lt2(x));
        
        Vpts = @(x) [ones(6,1) pts(x)' piplus(x)']*V(:,t+1);
        
        E1 = @(x) [.185 .63 .185 0 0 0]*Vpts(x);
        E2 = @(x) [0 0 0 .185 .63 .185]*Vpts(x);
        
        Vdum = @(x) U(x,S) + dlta*(P*E1(x)+(1-P)*E2(x));
        
        
        % i guess i don't need this loop for the regressions??
%         for k = 1:Nlt            
%             %U = bta*lt(k) - S^2;
%             
%             % do EV calculation
%             m1 = gmma*S + b + lt(k) + (S>Pcrit1)*r;
%             p5_1 = m1+pct5;
%             p95_1 = m1+pct95;
%             m2 = gmma*S + b + lt(k) + (S>Pcrit2)*r;
%             p5_2 = m2+pct5;
%             p95_2 = m2+pct95;
%             pts = [p5_1 m1 p95_1 p5_2 m2 p95_2];
%             
%             % likelihood functions and Bayesian updating
%             Lt1 = exp(-(pts - pts(2)).^2/(2*sgma^2));
%             Lt2 = exp(-(pts - pts(5)).^2/(2*sgma^2));
%             piplus = P*Lt1./(P*Lt1 + (1-P)*Lt2);
%             
%             % do interpolation for Vtp1
%             Vpts = interp2(X,Y,squeeze(V(:,:,t+1)),piplus,pts);
%             
%             E1 = .185*Vpts(1)+.63*Vpts(2)+.185*Vpts(3);
%             E2 = .185*Vpts(4)+.63*Vpts(5)+.185*Vpts(6);
%             
%             Vdum(k) = U + dlta*(P*E1+(1-P)*E2);
%         end
        if n <= Nr                          % make a random decision
            ltdum = rand*.8;
%             Vnthelp = Vdum(~isnan(Vdum));   % make sure it isn't NaN
%             Vnthelp2 = randperm(length(Vnthelp));
%             Vnt = Vnthelp(Vnthelp2(1));
        else
            ltdum = fminbnd(Vdum,0,.8);
        end
        Vnt = Vdum(ltdum);
        %ltdum = lt(Vnt==Vdum);
        
        % now figure this out!!
        
        
        
        V(S==Pt,P==pii,t) = Vnt;
        ltopt(S==Pt,P==pii,t) = ltdum;
        Sdum = gmma*S + b + ltdum + P*r*(S>Pcrit1) + (1-P)*r*(S>Pcrit2) + randn*sgma;
        
        Lt1b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit1)*r))^2/(2*sgma^2));
        Lt2b = exp(-(Sdum - (gmma*S + b + ltdum + (S>Pcrit2)*r))^2/(2*sgma^2));
        Pdum = P*Lt1b/(P*Lt1b + (1-P)*Lt2b);
        if Sdum < 0
            S = 0;
        elseif Sdum > 1
            S = 1;
        else
            S = interp1(Pt,Pt,Sdum,'nearest');
        end
        P = interp1(pii,pii,Pdum,'nearest');
    end
end

results.ltopt = ltopt;
results.V = V;
results.Pt = Pt;
results.lt = lt;
results.pii = pii;

end
        
        
        
        
        
        
        
        
        
        