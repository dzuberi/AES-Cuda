using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Threading;
using System.Windows.Threading;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Text.RegularExpressions;
using System.Text;

namespace gui
{
    public partial class Form1 : Form
    {
        private string inputFileName = "";
        private string outFileName = "";
        private int numThreads = 1;
        private bool enableCuda = false;
        private string aesKey = "                                ";
        private int aesKeyLength = 32;
        Task cppTask = Task.CompletedTask;
        Task timerTask = Task.CompletedTask;
        public static class aesWrapper
        {
            private const string DllFilePath = "..\\..\\..\\cuda\\aes.dll";

            [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
            private extern static int test(int number);
            [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
            static public extern IntPtr createNewWithOF(string iFilePath, string oFilePath, bool encrypt, int numThreads, bool cuda, string key, int keyLength);
            [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
            static public extern void destroy(IntPtr obj);

            public static void runAes(string iFilePath, string oFilePath, bool encrypt, int numThreads, bool cuda, string key, int keyLength)
            {
                IntPtr o = aesWrapper.createNewWithOF(iFilePath, oFilePath, encrypt, numThreads, cuda, key, keyLength);
                aesWrapper.destroy(o);
            }

        }

        public Form1()
        {
            InitializeComponent();
            lblHelloWorld.TextAlign = ContentAlignment.MiddleCenter;
            inputLabel.TextAlign = ContentAlignment.TopCenter;
            outputLabel.TextAlign = ContentAlignment.TopCenter;
            lblHelloWorld.Text = "Not started";
        }

        private void runAes(string iFilePath, string oFilePath, bool encrypt, int numThreads, bool cuda, string key, int keyLength)
        {
            aesWrapper.runAes(iFilePath, oFilePath, encrypt, numThreads, cuda, key, keyLength);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (cppTask.Status.Equals(TaskStatus.Running))
            {
                return;
            }
            lblHelloWorld.Text = "Running...";
            string outFileLocal = outFileName;
            if (!inputFileName.Equals("") && !inputFileName.Equals(outFileLocal))
            {
                if (outFileLocal == "")
                {
                    outFileLocal = inputFileName + ".aes";
                }
                var timer = System.Diagnostics.Stopwatch.StartNew();

                cppTask = Task.Run(()=>
                    { 
                        runAes(inputFileName, outFileLocal, radioEncrypt.Checked, numThreads, enableCuda, aesKey, aesKeyLength);
                        timer.Stop();
                    }
                );
                string elapsedMs;
                Dispatcher d = Dispatcher.CurrentDispatcher;
                cppTask.GetAwaiter().OnCompleted(() =>
                    {
                        d.Invoke((Action)delegate () {
                            elapsedMs = timer.ElapsedMilliseconds.ToString();
                            lblHelloWorld.Text = "Completed in " + elapsedMs + " ms";
                        });
                    }
                );
                //this.clearFiles();
            }
            else if (inputFileName.Equals(outFileLocal))
            {
                lblHelloWorld.Text = "The output file cannot be the same as the input file";
            }
            else
            {
                lblHelloWorld.Text = "No input file selected";
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Title = "Choose input file";
            DialogResult result = dialog.ShowDialog();
            if (result == DialogResult.OK) {
                inputLabel.Text = "" + dialog.FileName + "";
                inputFileName = dialog.FileName;
            }
            Console.WriteLine(result);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.Title = "Choose input file";
            DialogResult result = dialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                outputLabel.Text = "" + dialog.FileName + "";
                outFileName = dialog.FileName;
            }
            Console.WriteLine(result);
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            string text = threadsBox.Text;
            int number;
            try{
                number = Convert.ToInt32(text);
                if(number < 1){
                    threadsLabel.Text = "Number of threads invalid, defaulting to 1";
                }
                else{
                    threadsLabel.Text = "Number of threads: " + number;
                    numThreads = number;
                }
            }
            catch (FormatException){
                threadsLabel.Text = "Input is invalid, defaulting to 1";
            }
            catch (OverflowException){
                threadsLabel.Text = "Input is too big for Int32";
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            enableCuda = checkBox1.Checked;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            this.clearFiles();
        }

        private void clearFiles()
        {
            inputFileName = "";
            outFileName = "";
            outputLabel.Text = "No output file chosen, using (input file name).aes";
            inputLabel.Text = "No Input File Chosen";
        }
        private void updateKey()
        {
            string rawKey = keyTextBox.Text;
            rawKey = Regex.Replace(rawKey, @"[^\u0000-\u007F]+", " "); //replace all unicode characters with space;
            aesKeyLength = (key32.Checked) ? 32 : (key16.Checked) ? 16 : 32;
            StringBuilder builder = new StringBuilder();
            int i = 0;
            foreach (var c in rawKey)
            {
                builder.Append(c);
                ++i; //number of char added so far
                if (i == aesKeyLength) break;
            }
            while (i < aesKeyLength)
            {
                builder.Append("x");
                ++i;
            }
            aesKey = builder.ToString();
        }

        private void keyTextBox_TextChanged(object sender, EventArgs e)
        {
            this.updateKey();
        }

        private void key32_CheckedChanged(object sender, EventArgs e)
        {
            this.updateKey();
        }

        private void key16_CheckedChanged(object sender, EventArgs e)
        {
            this.updateKey();
        }
    }
}
