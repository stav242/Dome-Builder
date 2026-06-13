import streamlit as st
import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import io

st.set_page_config(page_title="Dome Builder", layout="wide")

class DomeGenerator:
    def generate_geodesic_sphere(self, radius, freq):
        phi = (1 + math.sqrt(5)) / 2
        verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ])
        # Normalize each vertex to unit length
        verts = np.array([v / np.linalg.norm(v) for v in verts])
        
        if freq > 1:
            new_verts = list(verts)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if np.linalg.norm(verts[i] - verts[j]) < 1.1: 
                        mid = (verts[i] + verts[j]) / 2.0
                        new_verts.append(mid / np.linalg.norm(mid))
            verts = np.array(new_verts)
            
        return verts * radius

    def poly_area_3d(self, vertices):
        n = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        n = n / np.linalg.norm(n)
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
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v1 = centered[0] / norm_0
        
        v2 = centered[1] - np.dot(centered[1], v1) * v1
        norm_2 = np.linalg.norm(v2)
        if norm_2 < 1e-10:
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v2 = v2 / norm_2
        
        coords_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in centered])
        return coords_2d

    def process_dome(self, radius, freq):
        centers = self.generate_geodesic_sphere(radius, freq)
        sv = SphericalVoronoi(centers, radius=radius, center=np.array([0, 0, 0]))
        sv.sort_vertices_of_regions()
        
        valid_regions = {}
        for i, region in enumerate(sv.regions):
            vertices = sv.vertices[region]
            centroid = np.mean(vertices, axis=0)
            if centroid[2] > -0.1:  
                valid_regions[i] = {'vertices': vertices, 'centroid': centroid, 'neighbors': {}}

        # Build neighbor relationships from ridge_points
        if hasattr(sv, 'ridge_points'):
            for i, j in sv.ridge_points:
                if i in valid_regions and j in valid_regions:
                    # Find shared edge vertices
                    region_i = sv.regions[i]
                    region_j = sv.regions[j]
                    shared_verts = np.intersect1d(region_i, region_j)
                    if len(shared_verts) >= 2:
                        shared_edge = sv.vertices[shared_verts]
                        valid_regions[i]['neighbors'][j] = shared_edge
                        valid_regions[j]['neighbors'][i] = shared_edge

        panel_types = []
        
        # Step 1: Assign Unique Instance IDs (For child papercraft assembly)
        # e.g., Panel 1, Panel 2, ..., Panel N
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
                if not np.allclose(p_type['edges'], edges_sorted, atol=1e-6, rtol=1e-4): continue
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
        page_width, page_height = 8.27, 11.69 # A4 dimensions in inches
        flap_depth = 1.5 # cm
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            for r_idx, data in valid_regions.items():
                fig, ax = plt.subplots(figsize=(page_width, page_height))
                ax.set_aspect('equal')
                ax.axis('off')
                
                pts_2d = self.to_2d(data['vertices'])
                poly = plt.Polygon(pts_2d, fill=True, color='#f5f5f5', ec='black', linestyle='--', linewidth=1.5)
                ax.add_patch(poly)
                
                # Use unique piece numbers for kids vs CNC shapes
                primary_label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
                
                # Main center ID
                ax.text(0, 0, primary_label, ha='center', va='center', fontsize=36, fontweight='bold', color='#222222')
                
                for j in range(len(pts_2d)):
                    p1 = pts_2d[j]
                    p2 = pts_2d[(j + 1) % len(pts_2d)]
                    edge_vec = p2 - p1
                    edge_len = np.linalg.norm(edge_vec)
                    
                    if edge_len < 1e-10:
                        continue
                    
                    edge_dir = edge_vec / edge_len
                    
                    normal = np.array([edge_dir[1], -edge_dir[0]])
                    midpoint = (p1 + p2) / 2.0
                    if np.dot(normal, midpoint) < 0: 
                        normal = -normal

                    # Tapered flaps to prevent collision when folded
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
                    
                    # Velcro target circle
                    velcro_dot = plt.Circle((text_pos[0], text_pos[1]), radius=0.45, fill=True, color='#ffffff', ec='#888888', linestyle=':')
                    ax.add_patch(velcro_dot)
                    
                    # Neighbor text placed perfectly over the velcro dot
                    ax.text(text_pos[0], text_pos[1], neighbor_id_text, ha='center', va='center', rotation=angle, fontsize=12, fontweight='bold', zorder=5)

                # Set hard limits so 1 coordinate unit = 1cm on A4 printout
                ax.set_xlim(-10.5, 10.5) 
                ax.set_ylim(-14.8, 14.8)
                
                mode_str = "Piece Number" if numbering_mode == "Instance" else "Panel Type"
                plt.title(f"Cut: Solid Line | Fold: Dashed Line\n{mode_str}: {primary_label} | (Target: Velcro Dot Guide)", fontsize=10)
                
                pdf.savefig(fig)
                plt.close(fig)
                
        pdf_buffer.seek(0)
        return pdf_buffer

    def create_3d_plot(self, valid_regions, numbering_mode="Instance"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Base colors on type_id even if numbering uniquely so pattern is visible
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
            
        collection = Poly3DCollection(faces, alpha=0.85, edgecolors='#333333', linewidths=0.8)
        collection.set_facecolor(face_colors)
        ax.add_collection3d(collection)
        
        for data in valid_regions.values():
            c = data['centroid']
            label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
            ax.text(c[0], c[1], c[2], label, 
                   color='black', fontsize=9, ha='center', va='center', 
                   zorder=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        all_points = np.vstack([data['vertices'] for data in valid_regions.values()])
        margin = 0.15
        range_vals = np.ptp(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        
        for i, (axis, setter) in enumerate([(ax.set_xlim, 'x'), (ax.set_ylim, 'y'), (ax.set_zlim, 'z')]):
            min_val = center[i] - range_vals[i] / 2 * (1 + margin)
            max_val = center[i] + range_vals[i] / 2 * (1 + margin)
            axis([min_val, max_val])
        
        ax.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (cm)', fontsize=10, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        fig.tight_layout()
        return fig

# --- UI LOGIC ---

st.title("🏛️ Goldberg Dome Papercraft & CNC Generator")
st.markdown("Generate custom geodesic dome plans and download printable PDFs for assembly.")

col1, col2, col3 = st.columns(3)

with col1:
    radius = st.number_input("Dome Radius (cm):", min_value=1.0, max_value=1000.0, value=15.0, step=1.0)
with col2:
    freq = st.number_input("Subdivision Frequency:", min_value=1, max_value=5, value=1, step=1)
with col3:
    num_mode = st.radio("Numbering System:", ["Unique Pieces (Papercraft)", "Panel Types (CNC)"])
    mode_arg = "Instance" if num_mode == "Unique Pieces (Papercraft)" else "Type"

if st.button("🔨 Generate Dome & Plans", type="primary", use_container_width=True):
    with st.spinner("⏳ Calculating geodesic geometry..."):
        generator = DomeGenerator()
        try:
            valid_regions, panel_types = generator.process_dome(radius, freq)
            
            # --- Safey Check for A4 Printing ---
            max_panel_width = 0
            for data in valid_regions.values():
                pts = generator.to_2d(data['vertices'])
                width = np.max(pts[:,0]) - np.min(pts[:,0])
                max_panel_width = max(max_panel_width, width)
            
            if max_panel_width + 3.0 > 21.0: # Panel width + 3cm buffer for flaps against 21cm A4
                st.warning(f"⚠️ **Warning**: At least one panel is {max_panel_width:.1f}cm wide. This will likely exceed the boundaries of standard A4 printer paper! Try reducing the dome radius or increasing subdivision frequency.", icon="📏")

            # Stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Physical Panels", len(valid_regions))
            with col2:
                st.metric("Unique CNC Shapes", len(panel_types))
            with col3:
                avg_panels = len(valid_regions) / len(panel_types) if panel_types else 0
                st.metric("Avg per Shape", f"{avg_panels:.1f}")
            with col4:
                dome_volume = (4/3) * np.pi * (radius ** 3) / 2
                st.metric("Volume (cm³)", f"{dome_volume:,.0f}")
            
            st.divider()
            
            st.markdown("### 📊 3D Preview")
            with st.spinner("🎨 Rendering 3D model..."):
                fig = generator.create_3d_plot(valid_regions, numbering_mode=mode_arg)
                st.pyplot(fig, use_container_width=True)
            
            st.divider()
            
            with st.spinner("📄 Compiling PDF with Flaps & Velcro Guides..."):
                pdf_buffer = generator.create_pdf_buffer(valid_regions, numbering_mode=mode_arg)
                
            st.markdown("### 📥 Download your Plans")
            st.info("The PDF maps exactly 1cm in the file to 1cm in reality. **Ensure you select 'Actual Size' or '100% Scale' when printing!**")
            
            st.download_button(
                label="📥 Download Printable Papercraft PDF",
                data=pdf_buffer,
                file_name=f"dome_plans_r{int(radius)}_f{freq}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {e}", icon="🚨")


```
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
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v1 = centered[0] / norm_0
        
        v2 = centered[1] - np.dot(centered[1], v1) * v1
        norm_2 = np.linalg.norm(v2)
        if norm_2 < 1e-10:
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v2 = v2 / norm_2
        
        coords_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in centered])
        return coords_2d

    def process_dome(self, radius, freq):
        centers = self.generate_geodesic_sphere(radius, freq)
        sv = SphericalVoronoi(centers, radius=radius, center=np.array([0, 0, 0]))
        sv.sort_vertices_of_regions()
        
        valid_regions = {}
        for i, region in enumerate(sv.regions):
            vertices = sv.vertices[region]
            centroid = np.mean(vertices, axis=0)
            if centroid[2] > -0.1:  
                valid_regions[i] = {'vertices': vertices, 'centroid': centroid, 'neighbors': {}}

        # Build neighbor relationships from ridge_points
        if hasattr(sv, 'ridge_points'):
            for i, j in sv.ridge_points:
                if i in valid_regions and j in valid_regions:
                    # Find shared edge vertices
                    region_i = sv.regions[i]
                    region_j = sv.regions[j]
                    shared_verts = np.intersect1d(region_i, region_j)
                    if len(shared_verts) >= 2:
                        shared_edge = sv.vertices[shared_verts]
                        valid_regions[i]['neighbors'][j] = shared_edge
                        valid_regions[j]['neighbors'][i] = shared_edge

        panel_types = []
        
        # Step 1: Assign Unique Instance IDs (For child papercraft assembly)
        # e.g., Panel 1, Panel 2, ..., Panel N
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
                if not np.allclose(p_type['edges'], edges_sorted, atol=1e-6, rtol=1e-4): continue
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
        page_width, page_height = 8.27, 11.69 # A4 dimensions in inches
        flap_depth = 1.5 # cm
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            for r_idx, data in valid_regions.items():
                fig, ax = plt.subplots(figsize=(page_width, page_height))
                ax.set_aspect('equal')
                ax.axis('off')
                
                pts_2d = self.to_2d(data['vertices'])
                poly = plt.Polygon(pts_2d, fill=True, color='#f5f5f5', ec='black', linestyle='--', linewidth=1.5)
                ax.add_patch(poly)
                
                # Use unique piece numbers for kids vs CNC shapes
                primary_label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
                
                # Main center ID
                ax.text(0, 0, primary_label, ha='center', va='center', fontsize=36, fontweight='bold', color='#222222')
                
                for j in range(len(pts_2d)):
                    p1 = pts_2d[j]
                    p2 = pts_2d[(j + 1) % len(pts_2d)]
                    edge_vec = p2 - p1
                    edge_len = np.linalg.norm(edge_vec)
                    
                    if edge_len < 1e-10:
                        continue
                    
                    edge_dir = edge_vec / edge_len
                    
                    normal = np.array([edge_dir[1], -edge_dir[0]])
                    midpoint = (p1 + p2) / 2.0
                    if np.dot(normal, midpoint) < 0: 
                        normal = -normal

                    # Tapered flaps to prevent collision when folded
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
                    
                    # Velcro target circle
                    velcro_dot = plt.Circle((text_pos[0], text_pos[1]), radius=0.45, fill=True, color='#ffffff', ec='#888888', linestyle=':')
                    ax.add_patch(velcro_dot)
                    
                    # Neighbor text placed perfectly over the velcro dot
                    ax.text(text_pos[0], text_pos[1], neighbor_id_text, ha='center', va='center', rotation=angle, fontsize=12, fontweight='bold', zorder=5)

                # Set hard limits so 1 coordinate unit = 1cm on A4 printout
                ax.set_xlim(-10.5, 10.5) 
                ax.set_ylim(-14.8, 14.8)
                
                mode_str = "Piece Number" if numbering_mode == "Instance" else "Panel Type"
                plt.title(f"Cut: Solid Line | Fold: Dashed Line\n{mode_str}: {primary_label} | (Target: Velcro Dot Guide)", fontsize=10)
                
                pdf.savefig(fig)
                plt.close(fig)
                
        pdf_buffer.seek(0)
        return pdf_buffer

    def create_3d_plot(self, valid_regions, numbering_mode="Instance"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Base colors on type_id even if numbering uniquely so pattern is visible
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
            
        collection = Poly3DCollection(faces, alpha=0.85, edgecolors='#333333', linewidths=0.8)
        collection.set_facecolor(face_colors)
        ax.add_collection3d(collection)
        
        for data in valid_regions.values():
            c = data['centroid']
            label = str(data['instance_id']) if numbering_mode == "Instance" else str(data['type_id'])
            ax.text(c[0], c[1], c[2], label, 
                   color='black', fontsize=9, ha='center', va='center', 
                   zorder=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        all_points = np.vstack([data['vertices'] for data in valid_regions.values()])
        margin = 0.15
        range_vals = np.ptp(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        
        for i, (axis, setter) in enumerate([(ax.set_xlim, 'x'), (ax.set_ylim, 'y'), (ax.set_zlim, 'z')]):
            min_val = center[i] - range_vals[i] / 2 * (1 + margin)
            max_val = center[i] + range_vals[i] / 2 * (1 + margin)
            axis([min_val, max_val])
        
        ax.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (cm)', fontsize=10, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        fig.tight_layout()
        return fig

# --- UI LOGIC ---

st.title("🏛️ Goldberg Dome Papercraft & CNC Generator")
st.markdown("Generate custom geodesic dome plans and download printable PDFs for assembly.")

col1, col2, col3 = st.columns(3)

with col1:
    radius = st.number_input("Dome Radius (cm):", min_value=1.0, max_value=1000.0, value=15.0, step=1.0)
with col2:
    freq = st.number_input("Subdivision Frequency:", min_value=1, max_value=5, value=1, step=1)
with col3:
    num_mode = st.radio("Numbering System:", ["Unique Pieces (Papercraft)", "Panel Types (CNC)"])
    mode_arg = "Instance" if num_mode == "Unique Pieces (Papercraft)" else "Type"

if st.button("🔨 Generate Dome & Plans", type="primary", use_container_width=True):
    with st.spinner("⏳ Calculating geodesic geometry..."):
        generator = DomeGenerator()
        try:
            valid_regions, panel_types = generator.process_dome(radius, freq)
            
            # --- Safey Check for A4 Printing ---
            max_panel_width = 0
            for data in valid_regions.values():
                pts = generator.to_2d(data['vertices'])
                width = np.max(pts[:,0]) - np.min(pts[:,0])
                max_panel_width = max(max_panel_width, width)
            
            if max_panel_width + 3.0 > 21.0: # Panel width + 3cm buffer for flaps against 21cm A4
                st.warning(f"⚠️ **Warning**: At least one panel is {max_panel_width:.1f}cm wide. This will likely exceed the boundaries of standard A4 printer paper! Try reducing the dome radius or increasing subdivision frequency.", icon="📏")

            # Stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Physical Panels", len(valid_regions))
            with col2:
                st.metric("Unique CNC Shapes", len(panel_types))
            with col3:
                avg_panels = len(valid_regions) / len(panel_types) if panel_types else 0
                st.metric("Avg per Shape", f"{avg_panels:.1f}")
            with col4:
                dome_volume = (4/3) * np.pi * (radius ** 3) / 2
                st.metric("Volume (cm³)", f"{dome_volume:,.0f}")
            
            st.divider()
            
            st.markdown("### 📊 3D Preview")
            with st.spinner("🎨 Rendering 3D model..."):
                fig = generator.create_3d_plot(valid_regions, numbering_mode=mode_arg)
                st.pyplot(fig, use_container_width=True)
            
            st.divider()
            
            with st.spinner("📄 Compiling PDF with Flaps & Velcro Guides..."):
                pdf_buffer = generator.create_pdf_buffer(valid_regions, numbering_mode=mode_arg)
                
            st.markdown("### 📥 Download your Plans")
            st.info("The PDF maps exactly 1cm in the file to 1cm in reality. **Ensure you select 'Actual Size' or '100% Scale' when printing!**")
            
            st.download_button(
                label="📥 Download Printable Papercraft PDF",
                data=pdf_buffer,
                file_name=f"dome_plans_r{int(radius)}_f{freq}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {e}", icon="🚨")

```
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
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v1 = centered[0] / norm_0
        
        v2 = centered[1] - np.dot(centered[1], v1) * v1
        norm_2 = np.linalg.norm(v2)
        if norm_2 < 1e-10:
            raise ValueError("Cannot create 2D projection: degenerate geometry")
        v2 = v2 / norm_2
        
        coords_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in centered])
        return coords_2d

    def process_dome(self, radius, freq):
        centers = self.generate_geodesic_sphere(radius, freq)
        sv = SphericalVoronoi(centers, radius=radius, center=np.array([0, 0, 0]))
        sv.sort_vertices_of_regions()
        
        valid_regions = {}
        for i, region in enumerate(sv.regions):
            vertices = sv.vertices[region]
            centroid = np.mean(vertices, axis=0)
            if centroid[2] > -0.1:  
                valid_regions[i] = {'vertices': vertices, 'centroid': centroid, 'neighbors': {}}

        # Build neighbor relationships from ridge_points (works with newer scipy)
        if hasattr(sv, 'ridge_points'):
            for i, j in sv.ridge_points:
                if i in valid_regions and j in valid_regions:
                    # Find shared edge vertices
                    region_i = sv.regions[i]
                    region_j = sv.regions[j]
                    shared_verts = np.intersect1d(region_i, region_j)
                    if len(shared_verts) >= 2:
                        shared_edge = sv.vertices[shared_verts]
                        valid_regions[i]['neighbors'][j] = shared_edge
                        valid_regions[j]['neighbors'][i] = shared_edge

        panel_types = []
        for i, data in valid_regions.items():
            verts = data['vertices']
            area = self.poly_area_3d(verts)
            edges = [np.linalg.norm(verts[k] - verts[(k+1)%len(verts)]) for k in range(len(verts))]
            edges_sorted = np.sort(edges)
            
            # Improved matching: check number of edges, edge lengths, and variance
            matched_type = -1
            for t_idx, p_type in enumerate(panel_types):
                # Must have same number of edges
                if len(p_type['edges']) != len(edges_sorted):
                    continue
                    
                # Check edge lengths with stricter tolerance
                if not np.allclose(p_type['edges'], edges_sorted, atol=1e-6, rtol=1e-4):
                    continue
                    
                # Check area similarity
                if abs(p_type['area'] - area) / max(p_type['area'], area) > 1e-4:
                    continue
                    
                # Check edge variance (handles geometric differences)
                var_match = np.var(p_type['edges'])
                var_current = np.var(edges_sorted)
                if abs(var_match - var_current) / max(abs(var_match), abs(var_current), 1e-10) > 1e-4:
                    continue
                    
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

    def create_pdf_buffer(self, valid_regions):
        page_width, page_height = 8.27, 11.69 
        flap_depth = 1.5 
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            for r_idx, data in valid_regions.items():
                fig, ax = plt.subplots(figsize=(page_width, page_height))
                ax.set_aspect('equal')
                ax.axis('off')
                
                pts_2d = self.to_2d(data['vertices'])
                poly = plt.Polygon(pts_2d, fill=True, color='#e0e0e0', ec='black', linestyle='--', linewidth=1.5)
                ax.add_patch(poly)
                
                ax.text(0, 0, str(data['type_id']), ha='center', va='center', fontsize=30, fontweight='bold', color='#333333')
                
                for j in range(len(pts_2d)):
                    p1 = pts_2d[j]
                    p2 = pts_2d[(j + 1) % len(pts_2d)]
                    edge_vec = p2 - p1
                    edge_len = np.linalg.norm(edge_vec)
                    
                    # Guard against zero-length edges
                    if edge_len < 1e-10:
                        continue
                    
                    edge_dir = edge_vec / edge_len
                    
                    normal = np.array([edge_dir[1], -edge_dir[0]])
                    midpoint = (p1 + p2) / 2.0
                    if np.dot(normal, midpoint) < 0: 
                        normal = -normal

                    inset = edge_len * 0.15
                    f1 = p1 + (edge_dir * inset) + (normal * flap_depth)
                    f2 = p2 - (edge_dir * inset) + (normal * flap_depth)
                    
                    flap_pts = np.array([p1, f1, f2, p2])
                    flap_poly = plt.Polygon(flap_pts, fill=False, ec='black', linestyle='-', linewidth=2)
                    ax.add_patch(flap_poly)
                    
                    neighbor_id_text = "BASE"
                    for n_idx, shared_3d_edge in data['neighbors'].items():
                        shared_len = np.linalg.norm(shared_3d_edge[0] - shared_3d_edge[1])
                        if abs(shared_len - edge_len) < 1e-4:
                            neighbor_id_text = str(valid_regions[n_idx]['type_id'])
                            break

                    text_pos = midpoint + (normal * (flap_depth * 0.6))
                    angle = np.degrees(np.arctan2(edge_dir[1], edge_dir[0]))
                    if angle > 90 or angle < -90: angle += 180 
                    
                    ax.text(text_pos[0], text_pos[1], neighbor_id_text, ha='center', va='center', rotation=angle, fontsize=12)

                ax.set_xlim(-10.5, 10.5) 
                ax.set_ylim(-14.8, 14.8)
                plt.title(f"Cut: Solid Line | Fold: Dashed Line\nPanel Type: {data['type_id']}", fontsize=10)
                
                pdf.savefig(fig)
                plt.close(fig)
                
        pdf_buffer.seek(0)
        return pdf_buffer

    def create_3d_plot(self, valid_regions):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate colors for different panel types
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
            
        collection = Poly3DCollection(faces, alpha=0.85, edgecolors='#333333', linewidths=0.8)
        collection.set_facecolor(face_colors)
        ax.add_collection3d(collection)
        
        # Add labels at centroids
        for data in valid_regions.values():
            c = data['centroid']
            ax.text(c[0], c[1], c[2], str(data['type_id']), 
                   color='black', fontsize=9, ha='center', va='center', 
                   zorder=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Auto-scale axes based on data range
        all_points = np.vstack([data['vertices'] for data in valid_regions.values()])
        margin = 0.15
        range_vals = np.ptp(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        
        for i, (axis, setter) in enumerate([(ax.set_xlim, 'x'), (ax.set_ylim, 'y'), (ax.set_zlim, 'z')]):
            min_val = center[i] - range_vals[i] / 2 * (1 + margin)
            max_val = center[i] + range_vals[i] / 2 * (1 + margin)
            axis([min_val, max_val])
        
        ax.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (cm)', fontsize=10, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        fig.tight_layout()
        return fig

# --- UI LOGIC ---
st.set_page_config(page_title="Dome Builder", layout="wide")

st.title("🏛️ Goldberg Dome CNC & Papercraft Generator")
st.markdown("Generate custom geodesic dome plans and download printable PDFs for assembly.")

col1, col2, col3 = st.columns(3)

with col1:
    radius = st.number_input("Dome Radius (cm):", min_value=1.0, max_value=1000.0, value=15.0, step=1.0)
    
with col2:
    freq = st.number_input("Subdivision Frequency:", min_value=1, max_value=5, value=1, step=1)
    
with col3:
    st.empty()

if st.button("🔨 Generate Dome & Plans", type="primary", use_container_width=True):
    with st.spinner("⏳ Calculating geodesic geometry..."):
        generator = DomeGenerator()
        try:
            valid_regions, panel_types = generator.process_dome(radius, freq)
            
            # Stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Panels", len(valid_regions))
            with col2:
                st.metric("Unique Types", len(panel_types))
            with col3:
                avg_panels = len(valid_regions) / len(panel_types) if panel_types else 0
                st.metric("Avg per Type", f"{avg_panels:.1f}")
            with col4:
                dome_volume = (4/3) * np.pi * (radius ** 3) / 2
                st.metric("Volume (cm³)", f"{dome_volume:,.0f}")
            
            st.divider()
            
            # 3D Visualization with full width
            st.markdown("### 📊 3D Preview")
            with st.spinner("🎨 Rendering 3D model..."):
                fig = generator.create_3d_plot(valid_regions)
                st.pyplot(fig, use_container_width=True)
            
            st.divider()
            
            # Panel type details table
            st.markdown("### 📋 Panel Type Details")
            panel_details = []
            for idx, p_type in enumerate(panel_types, 1):
                panel_details.append({
                    "Type": idx,
                    "Count": p_type['count'],
                    "Area (cm²)": f"{p_type['area']:.2f}",
                    "Edge Variance": f"{p_type['variance']:.6f}",
                    "Edges": f"{len(p_type['edges'])}"
                })
            
            st.dataframe(panel_details, use_container_width=True)
            
            st.divider()
            
            with st.spinner("📄 Compiling PDF..."):
                pdf_buffer = generator.create_pdf_buffer(valid_regions)
                
            st.markdown("### 📥 Download")
            st.download_button(
                label="📥 Download Printable Papercraft PDF",
                data=pdf_buffer,
                file_name=f"dome_plans_r{int(radius)}_f{freq}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {e}", icon="🚨")
