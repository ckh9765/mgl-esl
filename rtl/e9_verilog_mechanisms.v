// E9 per-tile mechanism cores for synthesis (yosys: `read_verilog e9_verilog_mechanisms.v; synth; stat`)
// 8-bit salience input p (quantised); thresholds tau_u/tau_b are 8-bit.
// 目的:比較三機制的 per-tile 狀態/邏輯資源。

// (a) TERNARY per-tile: 2-bit saturating counter + 2 threshold comparators
module ternary_tile #(parameter N=2) (
  input clk, rstn,
  input [7:0] p, tau_u, tau_b,
  output reg admit,      // 直送(p>=tau_u)
  output reg promote     // 連續 N 幀 >=tau_b 後放行
);
  reg [1:0] cnt;                       // 2-bit saturating counter(N<=3)
  wire above_u = (p >= tau_u);
  wire above_b = (p >= tau_b);
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin cnt<=0; admit<=0; promote<=0; end
    else begin
      admit   <= above_u;
      if(above_b && !above_u) cnt <= (cnt==2'd3)?2'd3:cnt+1'b1;  // defer band 累積
      else                    cnt <= 2'd0;
      promote <= (cnt >= N[1:0]) && above_b;
    end
  end
endmodule

// (b) SMOOTHING per-tile: W=5 x 8-bit shift buffer + running sum + 1 comparator
module smooth_tile #(parameter W=5) (
  input clk, rstn,
  input [7:0] p, tau_u,
  output reg admit
);
  reg [7:0] buf0,buf1,buf2,buf3,buf4;  // 5 x 8-bit = 40 FF
  reg [10:0] rsum;                     // running sum(11-bit)
  wire [10:0] avg5 = rsum / W;         // 除法(合成後為常數除,仍占面積)
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin buf0<=0;buf1<=0;buf2<=0;buf3<=0;buf4<=0;rsum<=0;admit<=0; end
    else begin
      rsum  <= rsum - buf4 + p;        // add new, subtract oldest
      buf4<=buf3; buf3<=buf2; buf2<=buf1; buf1<=buf0; buf0<=p;
      admit <= (avg5 >= tau_u);
    end
  end
endmodule

// (c) HYSTERESIS+timeout per-tile: 1-bit state + ceil(log2 T)-bit counter + 2 comparators
module hyst_tile #(parameter T=5, parameter CW=3) ( // CW=ceil(log2 T)
  input clk, rstn,
  input [7:0] p, tau_u, tau_b,
  output reg admit
);
  reg state;                           // 1-bit IDLE/ADMITTED
  reg [CW-1:0] cnt;                    // dwell counter
  wire above_u = (p >= tau_u);
  wire below_b = (p <  tau_b);
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin state<=0; cnt<=0; admit<=0; end
    else begin
      if(!state) begin
        if(above_u) begin state<=1; cnt<=0; end
      end else begin
        if(below_b || cnt>=T[CW-1:0]) state<=0;   // exit: drop or timeout
        else cnt<=cnt+1'b1;
      end
      admit <= state || (!state && above_u);
    end
  end
endmodule
