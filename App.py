import streamlit as st
import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import io

st.set_page_config(page_title="Parametric Dome Designer", layout="wide", page_icon="🏛️")

class DomeGenerator:
    def generate_geodesic_sphere(self, radius, freq):
        """
        Generates a Class I Geodesic Sphere using true barycentric subdivision.
        Incorporates the Onshape logic to pre-rotate the Icosahedron so a pentagon is top-dead-center.
        """
        phi = (1 + math.sqrt(5)) / 2.0
        
        # Base Icosahedron vertices
        verts = [
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ]
        verts = [np.array(v) / np.linalg.norm(v) for v in verts]

        # --- ONSHAPE FEATURESCRIPT LOGIC: Apex Alignment ---
        # Rotate the geometry around the X-axis so that the vertex at [0, 1, phi] 
        # (which becomes a pentagon) is perfectly aligned with the Z-axis [0, 0, 1].
        L = math.sqrt(1 + phi**2)
        cos_a = phi / L
        sin_a = 1 / L
        Rx = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        verts = [np.dot(v, Rx.T) for v in verts]

        # Standard Icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Barycentric subdivision
        subdivided_verts = []
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            for i in range(freq + 1):
                for j in range(freq + 1 - i):
                    k = freq - i - j
                    p = (i * v0 + j * v1 + k * v2) / freq
                    p = p / np.linalg.norm(p)
                    subdivided_verts.append(p)

        # Remove duplicates caused by shared edges/vertices
        unique_verts = []
        seen = set()
        for v in subdivided_verts:
            # Rounding for floating point safety when hashing
            t = tuple(np.round(v, 6))
            if t not in seen:
                seen.add(t)
                unique_verts.append(v)

        return np.array(unique_verts) * radius

    def poly_area_3d(self, vertices):
        n = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10: return 0.0
        n = n / n_norm
        area = 0.0
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            area += np.dot(np.cross(v1, v2), n)
        return abs(area) / 2.0

    def to_2d(self, vertices):
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        
        norm_0 = np.linalg.norm(centered[0])
        if norm_0 < 1e-10:
            raise ValueError("Degenerate geometry")
        v1 = centered[0] / norm_0
        
        v2 = centered[1] - np.dot(centered[1], v1) * v1
        norm_2 = np.linalg.norm(v2)
        if norm_2 < 1e-10:
            raise ValueError("Degenerate geometry")
        v2 = v2 / norm_2
        
        coords_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in centered])
        return coords_2d

    def process_dome(self, radius, freq, dome_height):
        # 1. Generate geometry
        centers = self.generate_geodesic_sphere(radius, freq)
        
        # Guard against minor floating point precision issues in scipy Voronoi
        centers = centers / np.linalg.norm(centers, axis=1)[:, np.newaxis] * radius
        
        sv = SphericalVoronoi(centers, radius=radius, center=np.array([0, 0, 0]))
        sv.sort_vertices_of_regions()
        
        # --- ONSHAPE FEATURESCRIPT LOGIC: Z-Cutoff ---
        # The top of the dome is at Z = +radius. 
        # A dome height of 'radius' implies a perfect hemisphere cut at Z = 0.
        z_cutoff = radius - dome_height

        valid_regions = {}
        for i, region in enumerate(sv.regions):
            vertices = sv.vertices[region]
            centroid = np.mean(vertices, axis=0)
            
            # Keep panels if their center is above the specified cutoff plane
            if centroid[2] >= z_cutoff:  
                valid_regions[i] = {'vertices': vertices, 'centroid': centroid, 'neighbors': {}}

        # Build neighbor relationships from ridge_points
        if hasattr(sv, 'ridge_points'):
            for i, j in sv.ridge_points:
                if i in valid_regions and j in valid_regions:
                    region_i = sv.regions[i]
                    region_j = sv.regions[j]
                    shared_verts = np.intersect1d(region_i, region_j)
                    if len(shared_verts) >= 2:
                        shared_edge = sv.vertices[shared_verts]
                        valid_regions[i]['neighbors'][j] = shared_edge
                        valid_regions[j]['neighbors'][i] = shared_edge

        panel_types = []
        
        # Step 1: Assign Unique Instance IDs 
        for inst_idx, (orig_id, data) in enumerate(valid_regions.items(), 1):
            data['instance_id'] = inst_idx

        # Step 2: Categorize by Master CNC Types (for BOM/Stats)
        for i, data in valid_regions.items():
            verts = data['vertices']
            area = self.poly_area_3d(verts)
            edges = [np.linalg.norm(verts[k] - verts[(k+1)%len(verts)]) for k in range(len(verts))]
            edges_sorted = np.sort(edges)
            
            matched_type = -1
            for t_idx, p_type in enumerate(panel_types):
                if len(p_type['edges']) != len(edges_sorted): continue
                if not np.allclose(p_type['edges'], edges_sorted, atol=1e-5, rtol=1e-4): continue
                if abs(p_type['area'] - area) / max(p_type['area'], area) > 1e-4: continue
                
                var_match = np.var(p_type['edges'])
                var_current = np.var(edges_sorted)
                if abs(var_match - var_current) / max(abs(var_match), abs(var_current), 1e-10) > 1e-4: continue
                    
                matched_type = t_idx
                break
            
            if matched_type == -1:
                panel_types.append({
                    'area': area, 
                    'edges': edges_sorted, 
                    'variance': np.var(edges_sorted),
                    'count': 1, 
                    'master_verts': verts
                })
                valid_regions[i]['type_id'] = len(panel_types)
            else:
                panel_types[matched_type]['count'] += 1
                valid_regions[i]['type_id'] = matched_type + 1

        return valid_regions, panel_types

    def create_pdf_buffer(self, valid_regions, numbering_mode="Instance"):
        page_width, page_height = 8.27, 11.69 # A4
        flap_depth = 1.5 # cm
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            for r_idx, data in valid_regions.items():
                fig, ax = plt.subplots(figsize=(page_width, page_height))
                ax.set_aspect('equal')
                ax.axis('off')
                
                pts_2d = self.to_2d(data['vertices'])
                poly = plt.Polygon(pts_2d, fill=True, color='#f8f9fa', ec='black', linestyle='--', linewidth=1.5)
                ax.add_patch(poly)
                
                primary_label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
                ax.text(0, 0, primary_label, ha='center', va='center', fontsize=36, fontweight='bold', color='#1a1a1a')
                
                for j in range(len(pts_2d)):
                    p1 = pts_2d[j]
                    p2 = pts_2d[(j + 1) % len(pts_2d)]
                    edge_vec = p2 - p1
                    edge_len = np.linalg.norm(edge_vec)
                    
                    if edge_len < 1e-10: continue
                    
                    edge_dir = edge_vec / edge_len
                    normal = np.array([edge_dir[1], -edge_dir[0]])
                    midpoint = (p1 + p2) / 2.0
                    
                    if np.dot(normal, midpoint) < 0: 
                        normal = -normal

                    inset = edge_len * 0.20
                    f1 = p1 + (edge_dir * inset) + (normal * flap_depth)
                    f2 = p2 - (edge_dir * inset) + (normal * flap_depth)
                    
                    flap_pts = np.array([p1, f1, f2, p2])
                    flap_poly = plt.Polygon(flap_pts, fill=False, ec='black', linestyle='-', linewidth=2)
                    ax.add_patch(flap_poly)
                    
                    neighbor_id_text = "BASE"
                    for n_idx, shared_3d_edge in data['neighbors'].items():
                        shared_len = np.linalg.norm(shared_3d_edge[0] - shared_3d_edge[1])
                        if abs(shared_len - edge_len) < 1e-4:
                            neighbor_id_text = str(valid_regions[n_idx]['instance_id']) if numbering_mode == "Instance" else str(valid_regions[n_idx]['type_id'])
                            break

                    text_pos = midpoint + (normal * (flap_depth * 0.5))
                    angle = np.degrees(np.arctan2(edge_dir[1], edge_dir[0]))
                    if angle > 90 or angle < -90: angle += 180 
                    
                    velcro_dot = plt.Circle((text_pos[0], text_pos[1]), radius=0.45, fill=True, color='#ffffff', ec='#aaaaaa', linestyle=':')
                    ax.add_patch(velcro_dot)
                    ax.text(text_pos[0], text_pos[1], neighbor_id_text, ha='center', va='center', rotation=angle, fontsize=12, fontweight='bold', zorder=5)

                ax.set_xlim(-10.5, 10.5) 
                ax.set_ylim(-14.8, 14.8)
                
                mode_str = "Piece Num" if numbering_mode == "Instance" else "Panel Type"
                plt.title(f"Cut: Solid Line | Fold: Dashed | {mode_str}: {primary_label} | Neighbor Maps to Target Dots", fontsize=9, color="#555")
                
                pdf.savefig(fig)
                plt.close(fig)
                
        pdf_buffer.seek(0)
        return pdf_buffer

    def create_3d_plot(self, valid_regions, numbering_mode="Instance"):
        fig = plt.figure(figsize=(10, 8), facecolor='none')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('none')
        
        panel_type_colors = {}
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        faces = []
        face_colors = []
        for data in valid_regions.values():
            faces.append(data['vertices'])
            type_id = data['type_id']
            if type_id not in panel_type_colors:
                panel_type_colors[type_id] = colors[(type_id - 1) % len(colors)]
            face_colors.append(panel_type_colors[type_id])
            
        collection = Poly3DCollection(faces, alpha=0.85, edgecolors='#4a4a4a', linewidths=0.6)
        collection.set_facecolor(face_colors)
        ax.add_collection3d(collection)
        
        for data in valid_regions.values():
            c = data['centroid']
            label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
            ax.text(c[0], c[1], c[2], label, 
                   color='black', fontsize=8, ha='center', va='center', 
                   zorder=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

        all_points = np.vstack([data['vertices'] for data in valid_regions.values()])
        margin = 0.15
        range_vals = np.ptp(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        
        for i, (axis, setter) in enumerate([(ax.set_xlim, 'x'), (ax.set_ylim, 'y'), (ax.set_zlim, 'z')]):
            min_val = center[i] - range_vals[i] / 2 * (1 + margin)
            max_val = center[i] + range_vals[i] / 2 * (1 + margin)
            axis([min_val, max_val])
        
        ax.axis('off')
        ax.view_init(elev=20, azim=-45)
        fig.tight_layout()
        return fig


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🏛️ Goldberg Dome Designer (Onshape Enabled)")
st.markdown("Parametric dome generation powered by Python & Onshape FeatureScript logic. Creates exportable documentation for CNC and papercraft prototyping.")

with st.sidebar:
    st.header("Dome Parameters")
    radius = st.number_input("Dome Radius (cm)", min_value=5.0, max_value=1000.0, value=15.0, step=1.0)
    
    st.markdown("---")
    
    st.markdown("**Dome Height (Z-Cutoff)**")
    height = st.number_input("Height from base to peak (cm)", 
                             min_value=float(radius * 0.1), 
                             max_value=float(radius * 2.0), 
                             value=float(radius), 
                             step=1.0,
                             help="Matches the FeatureScript 'Height' logic. Radius = Hemisphere.")
                             
    st.markdown("---")
    
    freq = st.slider("Subdivision Frequency (V)", min_value=1, max_value=5, value=2, 
                     help="Higher values create a smoother dome with more panels. Maximum recommended for A4 paper is V3.")
                     
    num_mode = st.radio("Documentation Mode:", ["Unique Pieces (Papercraft)", "Panel Types (CNC)"])
    mode_arg = "Instance" if num_mode == "Unique Pieces (Papercraft)" else "Type"
    
    generate = st.button("🔨 Generate Model", type="primary", use_container_width=True)


if generate:
    with st.spinner("⏳ Compiling parametric geometry..."):
        generator = DomeGenerator()
        try:
            valid_regions, panel_types = generator.process_dome(radius, freq, height)
            
            if not valid_regions:
                st.error("Height is too low, no panels generated above the cutoff plane.")
                st.stop()
            
            max_panel_width = 0
            for data in valid_regions.values():
                pts = generator.to_2d(data['vertices'])
                width = np.max(pts[:,0]) - np.min(pts[:,0])
                max_panel_width = max(max_panel_width, width)
            
            if max_panel_width + 3.0 > 21.0: 
                st.warning(f"⚠️ **Scale Warning**: At least one panel is {max_panel_width:.1f}cm wide. This may exceed A4 paper boundaries. Consider reducing the radius or increasing subdivision frequency.", icon="📏")

            # Dashboard Metrics
            cols = st.columns(4)
            cols[0].metric("Total Assembled Panels", len(valid_regions))
            cols[1].metric("Unique CNC Shapes", len(panel_types))
            cols[2].metric("Largest Panel Width", f"{max_panel_width:.1f} cm")
            volume = (4/3) * np.pi * (radius ** 3) * (height / (2*radius)) # Rough approx
            cols[3].metric("Est. Volume (cm³)", f"{volume:,.0f}")
            
            st.markdown("---")

            col_view, col_data = st.columns([2, 1])
            
            with col_view:
                st.markdown("### Interactive 3D Preview")
                fig = generator.create_3d_plot(valid_regions, numbering_mode=mode_arg)
                st.pyplot(fig, use_container_width=True)
                
            with col_data:
                st.markdown("### Manufacturing Specs")
                panel_details = []
                for idx, p_type in enumerate(panel_types, 1):
                    panel_details.append({
                        "Shape ID": idx,
                        "Qty Required": p_type['count'],
                        "Area (cm²)": f"{p_type['area']:.1f}",
                        "Edges": len(p_type['edges'])
                    })
                st.dataframe(panel_details, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            with st.spinner("📄 Generating Vector PDF Maps..."):
                pdf_buffer = generator.create_pdf_buffer(valid_regions, numbering_mode=mode_arg)
                
            st.success("Documentation ready! 1 cm in CAD = 1 cm on paper.")
            st.download_button(
                label="📥 Download Printable Assembly PDF",
                data=pdf_buffer,
                file_name=f"dome_v{freq}_r{int(radius)}_h{int(height)}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Geometric generation error: {e}", icon="🚨")
else:
    # Idle State
    st.info("👈 Adjust parameters in the sidebar and click **Generate Model** to start.")
