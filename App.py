import streamlit as st
import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import io

st.set_page_config(page_title="Dome Builder", layout="centered")

class DomeGenerator:
    def generate_geodesic_sphere(self, radius, freq):
        phi = (1 + math.sqrt(5)) / 2
        verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ])
        verts = verts / np.linalg.norm(verts[0])
        
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
        
        v1 = centered[0] / np.linalg.norm(centered[0])
        v2 = centered[1] - np.dot(centered[1], v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        
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

        for (i, j), ridge_vertices in sv.ridge_dict.items():
            if i in valid_regions and j in valid_regions:
                shared_edge = sv.vertices[ridge_vertices]
                valid_regions[i]['neighbors'][j] = shared_edge
                valid_regions[j]['neighbors'][i] = shared_edge

        panel_types = []
        for i, data in valid_regions.items():
            verts = data['vertices']
            area = self.poly_area_3d(verts)
            edges = [np.linalg.norm(verts[k] - verts[(k+1)%len(verts)]) for k in range(len(verts))]
            edges_sorted = np.sort(edges)
            
            matched_type = -1
            for t_idx, p_type in enumerate(panel_types):
                if abs(p_type['area'] - area) < 1e-3 and np.allclose(p_type['edges'], edges_sorted, atol=1e-3):
                    matched_type = t_idx
                    break
            
            if matched_type == -1:
                panel_types.append({'area': area, 'edges': edges_sorted, 'count': 1, 'master_verts': verts})
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
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        faces = []
        for data in valid_regions.values():
            faces.append(data['vertices'])
            
        collection = Poly3DCollection(faces, alpha=0.7, edgecolors='k', linewidths=1)
        collection.set_facecolor('#4CAF50')
        ax.add_collection3d(collection)
        
        for data in valid_regions.values():
            c = data['centroid']
            ax.text(c[0], c[1], c[2], str(data['type_id']), color='black', fontsize=10, ha='center', va='center', zorder=10)

        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.set_zlim([0, 15])
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        
        return fig

# --- UI LOGIC ---
st.title("Goldberg Dome CNC & Papercraft Generator")
st.markdown("Generate custom dome plans and download the printable PDF for assembly.")

radius = st.number_input("Dome Radius (cm):", min_value=1.0, max_value=1000.0, value=15.0, step=1.0)
freq = st.number_input("Subdivision Frequency (1 or 2 recommended):", min_value=1, max_value=5, value=1, step=1)

if st.button("Generate Dome & Plans", type="primary"):
    with st.spinner("Calculating geometry..."):
        generator = DomeGenerator()
        try:
            valid_regions, panel_types = generator.process_dome(radius, freq)
            
            st.success(f"Dome generated! Found {len(panel_types)} unique panel types out of {len(valid_regions)} total panels.")
            
            st.markdown("### 3D Preview")
            fig = generator.create_3d_plot(valid_regions)
            st.pyplot(fig)
            
            with st.spinner("Compiling PDF..."):
                pdf_buffer = generator.create_pdf_buffer(valid_regions)
                
            st.markdown("### Download")
            st.download_button(
                label="📥 Download Printable Papercraft PDF",
                data=pdf_buffer,
                file_name=f"dome_plans_r{int(radius)}_f{freq}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
