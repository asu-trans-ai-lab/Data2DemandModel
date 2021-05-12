classdef CPath<handle
    properties
        path_id
        from_zone_id
        to_zone_id
        K
        nb_node
        nb_link
        node_sequence
        link_sequence
        pass_link_vector
        path_cost
        path_travel_time
        path_estimated_proportion
        path_estimated_flow
        link_path_error_signal 
        path_theta_numerator_error_signal
        path_theta_dominator_error_signal
    end    
    methods 
        function obj=CPath(i,nb_link) % construct object path
            obj.path_id=i;
            obj.pass_link_vector=zeros(1,nb_link);
        end
        function obj= node2link(obj,node_sequence)
            [obj.nb_node,~]=size(node_sequence);
            obj.nb_link=obj.nb_node-1;
            obj.link_sequence=zeros(obj.nb_link,2);
            for ii=obj.nb_link:-1:1
                obj.link_sequence(ii,:)=[node_sequence(ii+1),node_sequence(ii)];
            end                    
        end
        function obj= Update_path_rtt(obj,g_link_vector)
            map_link_name_id=Hush_link_name_id(g_link_vector);% 建立一张hush table
            for k=1:obj.nb_link
                link_id=map_link_name_id(num2str(obj.link_sequence(k,:)));            
                obj.pass_link_vector(link_id)=1;
                obj.path_cost=obj.path_cost+g_link_vector(link_id).toll;
                obj.path_travel_time=obj.path_travel_time+g_link_vector(link_id).real_travel_time;
            end
        end
    end
end

