function [g_od_vector,g_path_vector,g_link_vector]=Forward_passing(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector,map_link_name_id)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);
%% ==========Step 1: forward passing from the o to od ==============
% Step 1.1 normalized for each iteration
% Step 1.2 generated estimated od volume for each iteration
for o=1:nb_ozone
    g_ozone_vector(o).total_gamma=0;
    for od=1:nb_od
        if  g_ozone_vector(o).ozone_id==g_od_vector(od).from_zone_id
            g_ozone_vector(o).total_gamma=g_ozone_vector(o).total_gamma...
                +g_od_vector(od).od_estimated_gamma; % we have generate rand gamma in TNode_Input.m file       
        end
    end
end

for od=1:nb_od
    for o=1:nb_ozone
        if g_ozone_vector(o).ozone_id==g_od_vector(od).from_zone_id
           % normalization the gamma
           g_od_vector(od).od_estimated_gamma=g_od_vector(od).od_estimated_gamma/g_ozone_vector(o).total_gamma;
           % calculate estimated demand using normalized gamma
           g_od_vector(od).estimated_demand= g_ozone_vector(o).estimated_generation*g_od_vector(od).od_estimated_gamma;
        end
   end
end
clear od;
clear o
fprintf('Forward passing from the layer of zones to the layer of od\n')
%% =========Step 2: forward passing from OD to path==================
% Step 2.1 using logit model to generate g_path_vector.path_estimated_proportion

[g_path_vector] = Logit(g_od_vector,g_path_vector);
%                 for p=1:nb_path
%                     g_path_vector(p)= Update_path_rtt(g_path_vector(p),g_link_vector);
%                 end

% Step 2.2 using logit model to generate g_path_vector.path_estimated_proportion
for p=1:nb_path
    for od=1:nb_od
        if g_od_vector(od).from_zone_id==g_path_vector(p).from_zone_id && g_od_vector(od).to_zone_id==g_path_vector(p).to_zone_id 
            g_path_vector(p).path_estimated_flow=g_od_vector(od).estimated_demand*g_path_vector(p).path_estimated_proportion;
         end
    end
end

clear p;
clear od
fprintf('Forward passing from the layer of od to the layer of path\n')
%%  ==============Step 3: forward passing the layer from path to link==================
for li=1:nb_link
    g_link_vector(li).estimated_flow=0;
    for p=1:nb_path
        if g_path_vector(p).pass_link_vector(g_link_vector(li).link_id)==1
           g_link_vector(li).estimated_flow=g_link_vector(li).estimated_flow+g_path_vector(p).path_estimated_flow;
        end
    end
    g_link_vector(li)=Update_link_rtt(g_link_vector(li)); % update real travel time using BPR function
end
fprintf('Forward passing from the layer of path to the layer of links and update their travel time\n')
clear li;
clear p

end