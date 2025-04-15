function plot_problem(problem,varargin)
    hold on
    A = problem.A; a = problem.supply;
    pos = problem.nodes;
    [m,n] = size(A);
    c = zeros(n,1); hw0 = 0.15; hh0 = 0.6;
    if isempty(varargin)
        for i=1:n
            x_from = find(A(:,i)==-1); 
            x_to = find(A(:,i)==1);
            if isempty(x_from) || isempty(x_to)
                continue;
            end
            plot([pos(x_from,1),pos(x_to,1)],[pos(x_from,2),pos(x_to,2)],'k');
        end
    else
       x = varargin{1};
       for i=1:n
            x_from = find(A(:,i)==-1); 
            x_to = find(A(:,i)==1);
            if isempty(x_from) || isempty(x_to) || x(i) < 1e-5
                continue;
            end
            c(i) = norm(pos(x_from,:)-pos(x_to,:));
            hw = hw0/c(i); hh = hh0/c(i);
            plot([pos(x_from,1),pos(x_to,1)],[pos(x_from,2),pos(x_to,2)],'k');
            text((pos(x_from,1)+pos(x_to,1))/2,(pos(x_from,2)+pos(x_to,2))/2,num2str(x(i)),'BackgroundColor','w','HorizontalAlignment','center');
       end
    end
    plot(pos(a>0,1),pos(a>0,2),'o','MarkerSize',5,'MarkerFaceColor',[0.5 0.5 1])
    plot(pos(a==0,1),pos(a==0,2),'o','MarkerSize',5,'MarkerFaceColor',[1 1 1])
    axis off;
end