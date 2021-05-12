classdef CLink<handle
    properties
      link_id
      from_node_id
      to_node_id
      length
      speed_limit
      free_flow_travel_time
      real_travel_time
      toll
      lane_cap
      BPR_alpha;
      BPR_belta;
      count_sensor_name
      estimated_flow
      target_flow

    end
    methods
        function obj=CLink()
        end
        % Calculate free flow travel time
        function obj=Calculate_FFTT(obj)
            obj.free_flow_travel_time=obj.length/obj.speed_limit;
        end
        % BPR function travel time updating
        function obj=Update_link_rtt(obj)
            obj.real_travel_time=obj.free_flow_travel_time*(1+obj.BPR_alpha*(obj.estimated_flow/obj.lane_cap)^obj.BPR_belta);
        end
    end
end

