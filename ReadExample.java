import java.io.FileInputStream;

public class ReadExample {
    public static void main(String[] args) {
        try {
            FileInputStream fis = new FileInputStream("a.txt");
            int i;

            while ((i = fis.read()) != -1) {
                System.out.print((char) i);
            }

            fis.close();
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}