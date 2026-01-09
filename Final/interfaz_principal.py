"""
Interfaz Principal Integrada - Sistema de Procesamiento de Imágenes
Acceso centralizado a todos los módulos del proyecto
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# Obtener directorio raíz del proyecto
DIRECTORIO_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Directorio del proyecto: {DIRECTORIO_PROYECTO}")


class InterfazPrincipal:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Integrado de Procesamiento de Imágenes")
        self.root.geometry("900x750")
        self.root.configure(bg='#2c3e50')
        
        # Centrar ventana
        self.centrar_ventana()
        
        # Crear interfaz
        self.crear_interfaz()
    
    def centrar_ventana(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        width = 900
        height = 750
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def crear_interfaz(self):
        """Crea la interfaz principal con acceso a todos los módulos"""
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título principal
        titulo = tk.Label(
            main_frame, 
            text="SISTEMA DE ANÁLISIS DE IMÁGENES",
            font=('Helvetica', 24, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        titulo.pack(pady=(10, 5))
        
        # Subtítulo
        subtitulo = tk.Label(
            main_frame,
            text="Seleccione el módulo que desea ejecutar",
            font=('Helvetica', 12),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        subtitulo.pack(pady=(0, 25))
        
        # Frame para botones
        botones_frame = tk.Frame(main_frame, bg='#2c3e50')
        botones_frame.pack(expand=True, fill='both', pady=10)
        
        # ===== BOTON 1: IMAGEN DIGITAL =====
        self.crear_boton(
            botones_frame,
            "1. Procesamiento Básico de Imágenes",
            "Conversión RGB→Grises, Binarización, Histogramas",
            '#3498db',
            lambda: self.abrir_modulo('ImagenDigital')
        )
        
        # ===== BOTON 2: OPERACIONES =====
        self.crear_boton(
            botones_frame,
            "2. Operaciones sobre Imágenes",
            "Operaciones aritméticas, lógicas y con escalares",
            '#e74c3c',
            lambda: self.abrir_modulo('Operaciones')
        )
        
        # ===== BOTON 3: SEGMENTACION =====
        self.crear_boton(
            botones_frame,
            "3. Técnicas de Segmentación",
            "Umbralización, Ecualización, Ajustes de histograma",
            '#9b59b6',
            lambda: self.abrir_modulo('Segmentacion')
        )
        
        # ===== BOTON 4: ANALISIS RUIDO =====
        self.crear_boton(
            botones_frame,
            "4. Análisis de Ruido y Filtros",
            "Generación de ruido, Filtros lineales y no lineales",
            '#f39c12',
            lambda: self.abrir_modulo('AnalisisRuido')
        )
        
        # ===== BOTON 5: MORFOLOGIA =====
        self.crear_boton(
            botones_frame,
            "5. Operaciones Morfológicas",
            "Erosión, Dilatación, Apertura, Cierre, Gradientes",
            '#1abc9c',
            lambda: self.abrir_modulo('Morfologia')
        )
        
        # ===== BOTON 6: FOURIER =====
        self.crear_boton(
            botones_frame,
            "6. Transformada de Fourier",
            "FFT, Filtros frecuenciales, Análisis DCT",
            '#e67e22',
            lambda: self.abrir_modulo('Fourier')
        )
        
        # ===== BOTON 7: HEATMAP =====
        self.crear_boton(
            botones_frame,
            "7. Mapas de Calor",
            "Generación y visualización de heatmaps",
            '#c0392b',
            lambda: self.abrir_modulo('HeatMap')
        )
        
        # Separador
        separador = tk.Frame(main_frame, bg='#34495e', height=2)
        separador.pack(fill='x', pady=15)
        
        # Botón de salir
        btn_salir = tk.Button(
            main_frame,
            text="SALIR DEL SISTEMA",
            command=self.salir,
            font=('Helvetica', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            width=25,
            height=2
        )
        btn_salir.pack(pady=10)
        
        # Footer
        footer = tk.Label(
            main_frame,
            text="© 2026 Sistema de Procesamiento de Imágenes",
            font=('Helvetica', 9),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        footer.pack(pady=(10, 0))
    
    
    def crear_boton(self, parent, titulo, descripcion, color, comando):
        """Crea un botón de módulo manualmente"""
        # Frame contenedor
        frame = tk.Frame(parent, bg='#34495e', relief='raised', bd=2)
        frame.pack(fill='x', pady=8, padx=10)
        
        # Frame interno
        frame_interno = tk.Frame(frame, bg='white')
        frame_interno.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Barra de color
        barra_color = tk.Frame(frame_interno, bg=color, height=5)
        barra_color.pack(fill='x')
        
        # Contenido
        contenido = tk.Frame(frame_interno, bg='white')
        contenido.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Título
        lbl_titulo = tk.Label(
            contenido,
            text=titulo,
            font=('Helvetica', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            anchor='w'
        )
        lbl_titulo.pack(fill='x')
        
        # Descripción
        lbl_desc = tk.Label(
            contenido,
            text=descripcion,
            font=('Helvetica', 9),
            bg='white',
            fg='#7f8c8d',
            anchor='w'
        )
        lbl_desc.pack(fill='x', pady=(2, 8))
        
        # Botón de acción
        btn = tk.Button(
            contenido,
            text="▶ ABRIR MÓDULO",
            command=comando,
            font=('Helvetica', 10, 'bold'),
            bg=color,
            fg='white',
            activebackground=self.oscurecer_color(color),
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=8
        )
        btn.pack(anchor='w')
        
        # Efecto hover
        btn.bind('<Enter>', lambda e: btn.configure(bg=self.oscurecer_color(color)))
        btn.bind('<Leave>', lambda e: btn.configure(bg=color))
    
    def oscurecer_color(self, hex_color):
        """Oscurece un color hexadecimal en un 20%"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, int(r * 0.75))
        g = max(0, int(g * 0.75))
        b = max(0, int(b * 0.75))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # ===== MÉTODOS DE APERTURA DE MÓDULOS =====
    
    def abrir_modulo(self, nombre_modulo):
        """Abre un módulo específico ejecutando su Main.py"""
        print(f"\n{'='*60}")
        print(f"Intentando abrir módulo: {nombre_modulo}")
        print(f"{'='*60}")
        
        try:
            # Ruta al Main.py del módulo
            if nombre_modulo == 'Morfologia':
                # Morfología usa interfaz_morfologia.py directamente
                main_path = os.path.join(DIRECTORIO_PROYECTO, nombre_modulo, "interfaz_morfologia.py")
            else:
                main_path = os.path.join(DIRECTORIO_PROYECTO, nombre_modulo, "Main.py")
            
            print(f"Ruta completa: {main_path}")
            
            # Verificar que existe el archivo
            if not os.path.exists(main_path):
                print(f"ERROR: No se encontró el archivo")
                messagebox.showerror(
                    "Error",
                    f"No se encontró el archivo:\n{main_path}"
                )
                return
            
            print(f"Archivo encontrado ✓")
            
            # Usar el Python del entorno virtual si existe
            venv_python = os.path.join(DIRECTORIO_PROYECTO, '.venv', 'Scripts', 'python.exe')
            if os.path.exists(venv_python):
                python_exe = venv_python
                print(f"Usando Python del entorno virtual: {python_exe}")
            else:
                python_exe = sys.executable
                print(f"Usando Python del sistema: {python_exe}")
            
            # Directorio de trabajo
            working_dir = os.path.join(DIRECTORIO_PROYECTO, nombre_modulo)
            print(f"Directorio de trabajo: {working_dir}")
            
            # Ejecutar el módulo
            print(f"Ejecutando módulo...")
            
            # Ejecutar sin capturar salida para que se muestre la ventana
            proceso = subprocess.Popen(
                [python_exe, main_path],
                cwd=working_dir
            )
            
            print(f"Proceso iniciado con PID: {proceso.pid}")
            print(f"Módulo {nombre_modulo} ejecutado exitosamente ✓")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            print(f"{'='*60}\n")
            messagebox.showerror(
                "Error al abrir módulo",
                f"No se pudo abrir el módulo {nombre_modulo}:\n\n{str(e)}"
            )
    
    def salir(self):
        """Cierra la aplicación"""
        if messagebox.askokcancel("Salir", "¿Está seguro de que desea salir del sistema?"):
            self.root.quit()
            self.root.destroy()


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("SISTEMA DE ANÁLISIS DE IMÁGENES - MENÚ PRINCIPAL")
    print("="*60 + "\n")
    
    root = tk.Tk()
    app = InterfazPrincipal(root)
    root.mainloop()


if __name__ == "__main__":
    main()
